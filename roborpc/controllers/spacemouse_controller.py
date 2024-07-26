import threading
import time
from typing import Union, Dict, List

import numpy as np
import pyspacemouse
from dataclasses import dataclass

from roborpc.cameras.transformations import apply_transfer
from roborpc.common.logger_loader import logger
from roborpc.controllers.controller_base import ControllerBase


@dataclass
class SpacemouseConfig:
    angle_scale: float = 0.24
    translation_scale: float = 0.06
    # only control the xyz, rotation direction, not the gripper
    invert_control: np.ndarray = np.ones(6)
    rotation_mode: str = "euler"


class SpaceMouseController(ControllerBase):
    def __init__(self, device_path: str):
        self.spacemouse = None
        self.last_state = None
        self.device_path = device_path
        self.config = SpacemouseConfig()
        self.last_state_lock = threading.Lock()
        spacemouse_thread = threading.Thread(target=self.run_spacemouse_thread)
        spacemouse_thread.start()

    def run_spacemouse_thread(self):
        while True:
            if self.spacemouse is not None:
                state = self.spacemouse.read()
                self.last_state = state
                print(state)
                time.sleep(0.001)
            else:
                time.sleep(1)

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        if self.device_path is None:
            self.spacemouse = pyspacemouse.open()
        else:
            self.spacemouse = pyspacemouse.open(path=self.device_path)
        if self.spacemouse is None:
            logger.error("Failed to connect to SpaceMouse")
            return False
        logger.info("Connected to SpaceMouse")
        return True

    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        return True

    def get_controller_id(self) -> List[str]:
        pass

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        return self.last_state

    def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[
        Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        obs_joints = obs_dict["joint_position"]
        obs_gripper = obs_dict["gripper_position"]

        spacemouse_state = self.last_state
        spacemouse_xyz_rpy = np.array([
            spacemouse_state.x,
            spacemouse_state.y,
            spacemouse_state.z,
            spacemouse_state.roll,
            spacemouse_state.picth,
            spacemouse_state.yaw
        ])
        spacemouse_xyz_rpy = spacemouse_xyz_rpy * self.config.invert_control
        if np.max(np.abs(spacemouse_xyz_rpy)) > 0.9:
            spacemouse_xyz_rpy[np.abs(spacemouse_xyz_rpy) < 0.6] = 0
        x, y, z, roll, pitch, yaw = spacemouse_xyz_rpy

        trans_transform = np.eye(4)
        spacemouse2ur = np.array(
            [
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        trans_transform[:3, 3] = apply_transfer(spacemouse2ur, np.array([x, y, z]) * self.config.translation_scale)



        return {}


if __name__ == "__main__":
    controller = SpaceMouseController(device_path="/dev/hidraw1")
    controller.connect_now()
    while True:
        time.sleep(0.1)
