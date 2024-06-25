import time
import threading
from typing import Dict, List, Union

import numpy as np
from pynput import keyboard
from pynput.keyboard import Key

from roborpc.controllers.controller_base import ControllerBase
from roborpc.robots.dynamixel import Dynamixel
from roborpc.common.logger_loader import logger


class DynamixelController(ControllerBase):

    def __init__(
            self,
            dynamixel: Dynamixel
    ):
        self.controller_id = None
        self.state = None
        self.robot = dynamixel
        self.gello_joints = None
        self.update_sensor = None
        self.move_start_num = None
        self.goto_start_pos = None
        self.move_start_flag = None
        self.key_button = False
        self.button_A_pressed = False
        self.button_B_pressed = False

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        result = self.robot.connect_now()
        self.controller_id = self.robot.robot_id
        self.reset_state()

        assert np.allclose(np.array(self.robot.get_robot_state()["robot_positions"]),
                           self.robot.start_joints, rtol=2, atol=2)
        threading.Thread(target=self.run_key_listen).start()
        threading.Thread(target=self._update_internal_state).start()
        time.sleep(1)
        return result

    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        return True

    def get_controller_id(self) -> List[str]:
        return [self.controller_id]

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
                self.button_A_pressed = True
            if key == Key.pause:
                self.button_B_pressed = True
        except AttributeError:
            logger.error(f"Unknown key {key}")

    def reset_state(self):
        self.state = {
            "gello_joints": {},
            "poses": {},
            "buttons": {"A": False, "B": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self.goto_start_pos = False
        self.move_start_flag = True
        self.move_start_num = 0

    def reset_gello_start_pos(self):
        self.goto_start_pos = True

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            time.sleep(1 / hz)

            time_since_read = time.time() - last_read_time
            dyna_joints = np.array(self.robot.get_robot_state()["robot_positions"])
            self.state["gello_joints"][self.controller_id] = dyna_joints
            self.state["controller_on"] = time_since_read < num_wait_sec
            self.update_sensor = True
            if self.button_A_pressed:
                print("button_A_pressed")
            if self.button_B_pressed:
                print("button_B_pressed")
            self.state["buttons"] = {"A": self.button_A_pressed, "B": self.button_B_pressed}
            self.state["movement_enabled"] = True
            self.state["controller_on"] = True
            last_read_time = time.time()

            self.button_A_pressed = False
            self.button_B_pressed = False

    def process_reading(self):
        gello_joints = np.asarray(self.state["gello_joints"][self.controller_id])
        self.gello_joints = {"gello_joints": gello_joints}

    def go_start_joints(self, state_dict):
        logger.info("Going to start joints")
        while self.state["gello_joints"] == {}:
            logger.info("No gello_joints yet, waiting...")
        start_pos = np.asarray(self.state["gello_joints"][self.controller_id])

        obs_joints = state_dict["joint_positions"]
        obs_gripper = state_dict["gripper_position"]
        if type(obs_gripper) == list:
            obs_gripper_new = obs_gripper[0]
        else:
            obs_gripper_new = obs_gripper
        obs_pos = np.concatenate([obs_joints, [obs_gripper_new]])

        abs_deltas = np.abs(start_pos - obs_pos)
        id_max_joint_delta = np.argmax(abs_deltas)

        max_joint_delta = 0.8
        if abs_deltas[id_max_joint_delta] > max_joint_delta:
            id_mask = abs_deltas > max_joint_delta
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                    ids,
                    abs_deltas[id_mask],
                    start_pos[id_mask],
                    obs_pos[id_mask],
            ):
                logger.info(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            return

        logger.info(f"Start pos: {len(start_pos)}", f"Joints: {len(obs_pos)}")
        assert len(start_pos) == len(
            obs_pos
        ), f"agent output dim = {len(start_pos)}, but env dim = {len(obs_pos)}"

        max_delta = 0.05
        self.move_start_num += 1
        if self.move_start_num == 24:
            self.move_start_flag = False
        if self.move_start_flag:
            command_joints = np.asarray(self.state["gello_joints"][self.controller_id])
            current_joints = np.concatenate([state_dict["joint_positions"], [state_dict["gripper_position"]]])
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            return current_joints + delta

        joints = np.concatenate([state_dict["joint_positions"], [state_dict["gripper_position"]]])
        action = np.asarray(self.state["gello_joints"][self.controller_id])
        if (action - joints > 0.5).any():
            logger.info("Joints are too big, exiting...")
            joint_index = np.where(action - joints > 0.5)
            for j in joint_index:
                logger.warning(
                    f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                )
            exit()
        self.goto_start_pos = True

    def calculate_action(self):
        # Read Sensor #
        if self.update_sensor:
            self.process_reading()
            self.update_sensor = False

        return np.asarray(self.state["gello_joints"][self.controller_id])

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        return {
            "success": self.state["buttons"]["A"],
            "failure": self.state["buttons"]["B"],
            "movement_enabled": self.state["movement_enabled"],
            "controller_on": self.state["controller_on"],
        }

    def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[List[float], Dict[str, List[float]]]:
        if not self.goto_start_pos:
            self.go_start_joints(obs_dict)
        return self.calculate_action().tolist()
