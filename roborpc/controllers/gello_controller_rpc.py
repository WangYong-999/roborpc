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

    def __init__(self, controller_id: str, ip_address: str, rpc_port: str = '4245'):
        self.controller_id = controller_id
        self.ip_address = ip_address
        self.rpc_port = rpc_port
        self.controller = None

    def connect_now(self):
        self.controller = zerorpc.Client(heartbeat=20)
        self.controller.connect_now("tcp://" + self.ip_address + ":" + self.rpc_port)

    def disconnect_now(self):
        pass

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        return self.controller.get_info()

    def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]):
        return self.controller.forward(obs_dict)


class MultiGelloControllerRpc(ControllerBase):
    def __init__(self):
        self.gello_controllers = {}

    def connect_now(self):
        robot_ids = self.robot_config["robot_ids"]
        self.robots = {}
        for idx, robot_id in enumerate(robot_ids):
            ip_address = self.robot_config["ip_address"][idx]
            rpc_port = self.robot_config["rpc_port"][idx]
            self.robots[robot_id] = RealManRpc(robot_id, ip_address, rpc_port)
            self.robots[robot_id].connect_now()
            logger.success(f"RealMan Robot {robot_id} Connect Success!")

    def disconnect_now(self):
        pass

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        info_dict = {}
        for controller_id in self.gello_controller_ids:
            info_dict[controller_id] = self.gello_controllers[controller_id].get_info()
        return info_dict

    def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]):
        for controller_id in self.gello_controller_ids:
            self.gello_controllers[controller_id].forward(obs_dict[controller_id])
