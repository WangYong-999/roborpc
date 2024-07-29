import threading
from typing import Union, Dict, List

import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
from roborpc.cameras.transformations import apply_transfer, euler_to_rmat, rmat_to_euler
from roborpc.common.logger_loader import logger

from roborpc.controllers.controller_base import ControllerBase


class QuestController(ControllerBase):
    def __init__(self, kinematic_solver=None, which_hand: str = "r"):
        super().__init__()
        self.kinematic_solver = kinematic_solver
        self.which_hand = which_hand
        self.button_A_pressed = False
        self.button_B_pressed = False
        self.quest2robot = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        self.ur2quest = np.linalg.inv(self.quest2robot)
        self.translation_scaling_factor = 2.0
        self.oculus_reader = None
        threading.Thread(target=self.run_key_listen).start()

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        from thirty_party.oculus_reader.oculus_reader.reader import OculusReader
        try:
            print("==============")
            self.oculus_reader = OculusReader()
        except Exception as e:
            logger.error(f"Failed to connect to Oculus reader: {e}")
            return False
        logger.success("Oculus reader connected successfully")
        return True

    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        pass

    def run_key_listen(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        try:
            if key == Key.scroll_lock:
                self.button_A_pressed = True
            if key == Key.pause:
                self.button_B_pressed = True
        except AttributeError:
            logger.error(f"Unknown key {key}")

    def on_release(self, key):
        try:
            if key == Key.scroll_lock:
                self.button_A_pressed = False
            if key == Key.pause:
                self.button_B_pressed = False
        except AttributeError:
            logger.error(f"Unknown key {key}")

    def get_controller_id(self) -> List[str]:
        pass

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        return {
            "success": self.button_A_pressed,
            "failure": self.button_B_pressed,
            "movement_enabled": True,
            "controller_on": True,
        }

    def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[
        Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        obs_joints = obs_dict["joint_position"]
        obs_gripper = obs_dict["gripper_position"]
        ee_pose = self.kinematic_solver.forward_kinematics(obs_joints)
        ee_rot = euler_to_rmat(ee_pose[3:])
        ee_pos = ee_pose[:3]

        pose_data, button_data = self.oculus_reader.get_transformations_and_buttons()
        # print(pose_data, button_data)
        if len(pose_data) == 0 or len(button_data) == 0:
            logger.warning("no data, quest not yet ready")
            return {"joint_position": obs_joints, "gripper_position": [0.0]}

        if self.which_hand == "l":
            pose_key = "l"
            trigger_key = "leftTrig"
            gripper_open_key = "Y"
            gripper_close_key = "X"
        elif self.which_hand == "r":
            pose_key = "r"
            trigger_key = "rightTrig"
            gripper_open_key = "B"
            gripper_close_key = "A"
        else:
            raise ValueError(f"Unknown hand: {self.which_hand}")

        new_gripper_angle = 0
        if button_data[gripper_open_key]:
            new_gripper_angle = 1
        if button_data[gripper_close_key]:
            new_gripper_angle = 0
        if len(pose_data) == 0:
            logger.warning("no data, quest not yet ready")
            return {"joint_position": obs_joints, "gripper_position": [new_gripper_angle]}

        trigger_state = button_data[trigger_key][0]
        if trigger_state > 0.5:
            if self.control_active is True:
                current_pose = pose_data[pose_key]
                delta_rot = current_pose[:3, :3] @ np.linalg.inv(
                    self.reference_quest_pose[:3, :3]
                )
                delta_pos = current_pose[:3, 3] - self.reference_quest_pose[:3, 3]
                delta_pos_robot = (
                        apply_transfer(self.quest2robot, delta_pos) * self.translation_scaling_factor
                )
                delta_rot_robot = self.quest2robot[:3, :3] @ delta_rot @ self.ur2quest[:3, :3]
                next_ee_rot_robot = rmat_to_euler(delta_rot_robot @ self.reference_ee_rot_robot)
                next_ee_pos_robot = delta_pos_robot + self.reference_ee_pos_robot

                return {"cartesian_position": np.concatenate([next_ee_pos_robot, next_ee_rot_robot]).tolist(),
                        "gripper_position": [new_gripper_angle]}
            else:
                self.control_active = True
                self.reference_quest_pose = pose_data[pose_key]
                self.reference_ee_rot_robot = ee_rot
                self.reference_ee_pos_robot = ee_pos
                return {"joint_position": obs_joints, "gripper_position": [new_gripper_angle]}
        else:
            self.control_active = False
            self.reference_quest_pose = None
            return {"joint_position": obs_joints, "gripper_position": [new_gripper_angle]}
