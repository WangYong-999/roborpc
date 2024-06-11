import os
import threading
import time
from typing import Optional, Sequence, Tuple, Dict, List, Union

import numpy as np
import serial
from dataclasses import dataclass
from pynput import keyboard
from pynput.keyboard import Key

from roborpc.controllers.controller_base import ControllerBase
from roborpc.robots.dynamixel import Dynamixel
from roborpc.common.config_loader import config as common_config
from roborpc.common.logger_loader import logger


class GelloController(ControllerBase):
    def __init__(
            self,
            dynamixel: Dynamixel
    ):
        self.gello_joints = None
        self.update_sensor = None
        self.move_start_num = None
        self._goto_start_pos = None
        self.move_start_flag = None
        self._state = None
        self.key_button = False
        self._robot = dynamixel
        logger.info("Gello Controller Initialized")
        self._robot.connect()
        self.controller_id = self._robot.robot_id
        self.reset_state()

        assert np.allclose(np.array(self._robot.get_robot_state()["robot_positions"]), self._robot.start_joints, rtol=2, atol=2)

        self.button_A_pressed = False
        self.button_B_pressed = False
        threading.Thread(target=self.run_key_listen).start()
        threading.Thread(target=self._update_internal_state).start()
        time.sleep(3)

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
            ...

    def on_release(self, key):
        try:
            if key == Key.scroll_lock:
                self.button_A_pressed = True
            if key == Key.pause:
                self.button_B_pressed = True
        except AttributeError:
            ...

    def reset_state(self):
        self._state = {
            "gello_joints": {},
            "poses": {},
            "buttons": {"A": False, "B": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self._goto_start_pos = False
        self.move_start_flag = True
        self.move_start_num = 0

    def reset_gello_start_pos(self):
        self._goto_start_pos = True

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            # Read Controller
            time_since_read = time.time() - last_read_time
            dyna_joints = np.array(self._robot.get_robot_state()["robot_positions"])
            # print(f"dyna_joints: {dyna_joints}")
            current_q = dyna_joints[:-1]  # last one dim is the gripper
            current_gripper = dyna_joints[-1]  # last one dim is the gripper
            # Save Info #
            self._state["gello_joints"][self.controller_id] = dyna_joints
            self._state["controller_on"] = time_since_read < num_wait_sec
            # Determine Control Pipeline #
            # poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            # button_A_data = self.button_A.readline()
            # button_B_data = self.button_B.readline()
            # if button_A_data == b'\x00':
            #     self.button_A_pressed = True
            # else:
            #     self.button_A_pressed = False
            #
            # if button_B_data == b'\x00':
            #     self.button_B_pressed = True
            # else:
            #     self.button_B_pressed = False
            self.update_sensor = True
            if self.button_A_pressed:
                print("button_A_pressed")
            if self.button_B_pressed:
                print("button_B_pressed")
            self._state["buttons"] = {"A": self.button_A_pressed, "B": self.button_B_pressed}
            self._state["movement_enabled"] = True
            self._state["controller_on"] = True
            last_read_time = time.time()

            self.button_A_pressed = False
            self.button_B_pressed = False

    def process_reading(self):
        gello_joints = np.asarray(self._state["gello_joints"][self.controller_id])
        self.gello_joints = {"gello_joints": gello_joints}

    def go_start_joints(self, state_dict):
        # going to start position
        print("Going to start position")
        # get gello data
        while self._state["gello_joints"] == {}:
            print("gello joints is empty")
        start_pos = np.asarray(self._state["gello_joints"][self.controller_id])

        # get obs data (Franka)
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
            print()
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                    ids,
                    abs_deltas[id_mask],
                    start_pos[id_mask],
                    obs_pos[id_mask],
            ):
                print(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            return

        print(f"Start pos: {len(start_pos)}", f"Joints: {len(obs_pos)}")
        assert len(start_pos) == len(
            obs_pos
        ), f"agent output dim = {len(start_pos)}, but env dim = {len(obs_pos)}"

        max_delta = 0.05
        self.move_start_num += 1
        if self.move_start_num == 24:
            self.move_start_flag = False
        if self.move_start_flag:
            command_joints = np.asarray(self._state["gello_joints"][self.controller_id])
            current_joints = np.concatenate([state_dict["joint_positions"], [state_dict["gripper_position"]]])
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            return current_joints + delta

        joints = np.concatenate([state_dict["joint_positions"], [state_dict["gripper_position"]]])
        action = np.asarray(self._state["gello_joints"][self.controller_id])
        if (action - joints > 0.5).any():
            print("Action is too big")

            # print which joints are too big
            joint_index = np.where(action - joints > 0.5)
            for j in joint_index:
                print(
                    f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                )
            exit()
        self._goto_start_pos = True

    def calculate_action(self):
        # Read Sensor #
        if self.update_sensor:
            self.process_reading()
            self.update_sensor = False

        return np.asarray(self._state["gello_joints"][self.controller_id])

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        return {
            "success": self._state["buttons"]["A"],
            "failure": self._state["buttons"]["B"],
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }

    def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]):
        if not self._goto_start_pos:
            self.go_start_joints(obs_dict["robot_state"])
        return self.calculate_action()


class MultiGelloController(ControllerBase):
    def __init__(self):
        self.gello_controllers = {}
        self.gello_controller_ids = common_config["roborpc"]["controllers"]["gello_controller"]["controller_ids"]
        self.robots_config = common_config["roborpc"]["robots"]["dynamixel"]
        for controller_id in self.gello_controller_ids:
            self.gello_controllers[controller_id] = GelloController(
                Dynamixel(
                    robot_id=controller_id,
                    joint_ids=self.robots_config[controller_id]["joint_ids"],
                    joint_signs=self.robots_config[controller_id]["joint_signs"],
                    joint_offsets=self.robots_config[controller_id]["joint_offsets"],
                    start_joints=self.robots_config[controller_id]["start_joints"],
                    port=self.robots_config[controller_id]["port"],
                    gripper_config=self.robots_config[controller_id]["gripper_config"]
                ),
            )

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        info_dict = {}
        for controller_id in self.gello_controller_ids:
            info_dict[controller_id] = self.gello_controllers[controller_id].get_info()
        return info_dict

    def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]):
        for controller_id in self.gello_controller_ids:
            self.gello_controllers[controller_id].forward(obs_dict[controller_id])
