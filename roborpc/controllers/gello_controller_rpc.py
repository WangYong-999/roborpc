import os
import threading
import time
from typing import Optional, Sequence, Tuple, Dict, List, Union
import zerorpc
import numpy as np
import serial
from dataclasses import dataclass
from pynput import keyboard
from pynput.keyboard import Key

from roborpc.controllers.controller_base import ControllerBase
from roborpc.common.config_loader import config as common_config
from roborpc.common.logger_loader import logger


class GelloControllerRpc(ControllerBase):
    def __init__(self, robot):
        super().__init__(robot)

    def connect(self):
        self.controller = zerorpc.Client(heartbeat=20)
        self.controller.connect("tcp://" + self.ip_address + ":" + self.rpc_port)

    def disconnect(self):
        pass

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


class MultiGelloControllerRpc(ControllerBase):
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
