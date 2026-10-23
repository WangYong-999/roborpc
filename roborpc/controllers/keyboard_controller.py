import threading
import time
from typing import Union, Dict, List, Optional

import numpy as np
from dataclasses import dataclass
import quaternion
from pynput import keyboard
from pynput.keyboard import Key

from roborpc.cameras.transformations import apply_transfer, rotation_matrix, euler_to_rmat
from roborpc.common.logger_loader import logger
from roborpc.controllers.controller_base import ControllerBase


@dataclass
class KeyboardConfig:
    angle_scale: float = 1.0
    translation_scale: float = 1.0
    # only control the xyz, rotation direction, not the gripper
    invert_control: np.ndarray = np.ones(6)
    rotation_mode: str = "rpy"

class KeyboardController(ControllerBase):
    def __init__(self, kinematic_solver=None, device_path: Optional[str] = None):
        self.kinematic_solver = kinematic_solver
        self.spacemouse = None
        self.last_state = None
        self.device_path = device_path
        self.config = KeyboardConfig()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self._pos_step = 0.05
        self.pos_sensitivity = self.config.translation_scale
        self.rot_sensitivity = self.config.angle_scale
        self.grasp = False

        self.toogle_key = False
        self.button_A_pressed = False
        self.button_B_pressed = False
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

            if key == Key.space:
                self.grasp = not self.grasp  # toggle gripper

            # controls for moving position
            if key == Key.up:
                self.pos[0] -= self._pos_step * self.pos_sensitivity  # dec x
                self.toogle_key = True
            elif key == Key.down:
                self.pos[0] += self._pos_step * self.pos_sensitivity  # inc x
                self.toogle_key = True
            elif key == Key.left:
                self.pos[1] -= self._pos_step * self.pos_sensitivity  # dec y
                self.toogle_key = True
            elif key == Key.right:
                self.pos[1] += self._pos_step * self.pos_sensitivity  # inc y
                self.toogle_key = True
            elif key.char == ".":
                self.pos[2] -= self._pos_step * self.pos_sensitivity  # dec z
                self.toogle_key = True
            elif key.char == ";":
                self.pos[2] += self._pos_step * self.pos_sensitivity  # inc z
                self.toogle_key = True
            # controls for moving orientation
            elif key.char == "e":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] -= 0.1 * self.rot_sensitivity
                self.toogle_key = True
            elif key.char == "r":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] += 0.1 * self.rot_sensitivity
                self.toogle_key = True
            elif key.char == "y":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] += 0.1 * self.rot_sensitivity
                self.toogle_key = True
            elif key.char == "h":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] -= 0.1 * self.rot_sensitivity
                self.toogle_key = True
            elif key.char == "p":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] += 0.1 * self.rot_sensitivity
                self.toogle_key = True
            elif key.char == "o":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] -= 0.1 * self.rot_sensitivity
                self.toogle_key = True
            else:
                self.toogle_key = False

        except AttributeError:
            logger.error(f"Unknown key {key}")

    def on_release(self, key):
        try:
            if key == Key.scroll_lock:
                self.button_A_pressed = False
            if key == Key.pause:
                self.button_B_pressed = False
            self.toogle_key = False
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

        if self.grasp:
            new_gripper_angle = 1.0
        else:
            new_gripper_angle = 0.0

        if not self.toogle_key:
            return {"joint_position": obs_joints, "gripper_position": [new_gripper_angle]}
        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = euler_to_rmat(self.raw_drotation - self.last_drotation)
        self.last_drotation = np.array(self.raw_drotation)

        new_ee_pos = dpos + ee_pose[:3]
        if self.config.rotation_mode == "rpy":
            new_ee_rot = ee_pose[3:] @ raw_drotation[:3, :3]
        elif self.config.rotation_mode == "euler":
            new_ee_rot = raw_drotation[:3, :3] @ ee_pose[3:]
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
