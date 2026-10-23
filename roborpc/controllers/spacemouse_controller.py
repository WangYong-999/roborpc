import threading
import time
from typing import Union, Dict, List, Optional

import numpy as np
from dataclasses import dataclass
import quaternion
from pynput import keyboard
from pynput.keyboard import Key

from roborpc.cameras.transformations import apply_transfer
from roborpc.common.logger_loader import logger
from roborpc.controllers.controller_base import ControllerBase


@dataclass
class SpacemouseConfig:
    angle_scale: float = 0.24
    translation_scale: float = 0.06
    # only control the xyz, rotation direction, not the gripper
    invert_control: np.ndarray = np.ones(6)
    rotation_mode: str = "rpy"


class SpaceMouseController(ControllerBase):
    def __init__(self, kinematic_solver=None, device_path: Optional[str] = None):
        self.kinematic_solver = kinematic_solver
        self.spacemouse = None
        self.last_state = None
        self.device_path = device_path
        self.config = SpacemouseConfig()
        self.spacemouse2robot = np.array(
            [
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.button_A_pressed = False
        self.button_B_pressed = False
        self.robot2spacemouse = np.linalg.inv(self.spacemouse2robot)
        self.last_state_lock = threading.Lock()
        spacemouse_thread = threading.Thread(target=self.run_spacemouse_thread)
        threading.Thread(target=self.run_key_listen).start()
        spacemouse_thread.start()

    def run_spacemouse_thread(self):
        while True:
            if self.spacemouse is not None:
                state = self.spacemouse.read()
                with self.last_state_lock:
                    self.last_state = state
                time.sleep(0.001)
            else:
                time.sleep(0.001)

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

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        import pyspacemouse
        if self.device_path is None:
            self.spacemouse = pyspacemouse.open()
        else:
            self.spacemouse = pyspacemouse.open()
        if self.spacemouse is None:
            logger.error("Failed to connect to SpaceMouse")
            return False
        logger.success("Connected to SpaceMouse")
        return True

    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        return True

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
        if self.last_state is None:
            logger.warning("No SpaceMouse state available")
            return {"joint_position": obs_joints, "gripper_position": [0.0]}
        with self.last_state_lock:
            spacemouse_state = self.last_state
        spacemouse_xyz_rpy = np.array([
            spacemouse_state.x,
            spacemouse_state.y,
            spacemouse_state.z,
            spacemouse_state.roll,
            spacemouse_state.pitch,
            spacemouse_state.yaw
        ])
        print(spacemouse_xyz_rpy)
        spacemouse_button = spacemouse_state.buttons
        new_gripper_angle = 0
        if spacemouse_button[1]:
            new_gripper_angle = 0
        if spacemouse_button[0]:
            new_gripper_angle = 1
        if np.all(np.abs(spacemouse_xyz_rpy) < 0.01):
            return {"joint_position": obs_joints, "gripper_position": [new_gripper_angle]}
        spacemouse_xyz_rpy = spacemouse_xyz_rpy * self.config.invert_control
        if np.max(np.abs(spacemouse_xyz_rpy)) > 0.9:
            spacemouse_xyz_rpy[np.abs(spacemouse_xyz_rpy) < 0.6] = 0
        x, y, z, roll, pitch, yaw = spacemouse_xyz_rpy

        trans_transform = np.eye(4)

        trans_transform[:3, 3] = apply_transfer(self.spacemouse2robot, np.array([x, y, z]) * self.config.translation_scale)

        rot_transform_x = np.eye(4)
        rot_transform_x[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(
                np.array([-pitch, 0, 0]) * self.config.angle_scale
            )
        )

        rot_transform_y = np.eye(4)
        rot_transform_y[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(
                np.array([0, roll, 0]) * self.config.angle_scale
            )
        )

        rot_transform_z = np.eye(4)
        rot_transform_z[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(
                np.array([0, 0, -yaw]) * self.config.angle_scale
            )
        )

        # in ur space
        rot_transform = (
                self.spacemouse2robot
                @ rot_transform_z
                @ rot_transform_y
                @ rot_transform_x
                @ self.robot2spacemouse
        )
        new_ee_pos = trans_transform[:3, 3] + ee_pose[:3]
        if self.config.rotation_mode == "rpy":
            new_ee_rot = ee_pose[3:] @ rot_transform[:3, :3]
        elif self.config.rotation_mode == "euler":
            new_ee_rot = rot_transform[:3, :3] @ ee_pose[3:]
        else:
            raise NotImplementedError(
                f"Unknown rotation mode: {self.config.rotation_mode}"
            )
        return {"cartesian_position": np.concatenate([new_ee_pos, new_ee_rot]).tolist(),
                "gripper_position": [new_gripper_angle]}


if __name__ == "__main__":
    controller = SpaceMouseController()
    controller.connect_now()
    while True:
        time.sleep(0.1)
