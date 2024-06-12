from typing import Tuple

import numpy as np

from droid.dynamixel.driver import DynamixelDriver
from dataclasses import dataclass
from common.config_loader import config as common_config


@dataclass
class Args:
    port: str = common_config["droid"]["robot"]["gello_port"]
    """The port that GELLO is connected to."""

    start_joints: Tuple[float, ...] = tuple(common_config["droid"]["robot"]["start_joints"][:-1])
    """The joint angles that the GELLO is placed in at (in radians)."""

    joint_signs: Tuple[float, ...] = (1, -1, 1, 1, 1, -1, 1)
    """The joint angles that the GELLO is placed in at (in radians)."""

    gripper: bool = True
    """Whether or not the gripper is attached."""

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert (
                j == -1 or j == 1
            ), f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_robot_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints


def get_config():
    args = Args()
    joint_ids = list(range(1, args.num_joints + 1))
    driver = DynamixelDriver(joint_ids, port=args.port, baudrate=57600)
    joint_offset_list = []
    gripper_offset_list = []

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = args.joint_signs[index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = args.start_joints[index]
        return np.abs(joint_i - start_i)

    for _ in range(10):
        driver.get_joints()  # warmup

    for _ in range(1):
        best_offsets = []
        curr_joints = driver.get_joints()
        for i in range(args.num_robot_joints):
            best_offset = 0
            best_error = 1e6
            for offset in np.linspace(
                -8 * np.pi, 8 * np.pi, 8 * 4 + 1
            ):  # intervals of pi/2
                error = get_error(offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            best_offsets.append(best_offset)
        joint_offset_list = [int(np.round(x / (np.pi / 2))) * np.pi / 2 for x in best_offsets]
        if args.gripper:
            gripper_offset_list = [np.rad2deg(driver.get_joints()[-1]) - 0.2, np.rad2deg(driver.get_joints()[-1]) - 42]

    print(f"joint_offset_list: {joint_offset_list}")
    print(f"gripper_offset_list: {gripper_offset_list}")
    # [1.5707963267948966, 3.141592653589793, 3.141592653589793, 4.71238898038469, 3.141592653589793, 4.71238898038469,
     # 1.5707963267948966]
    # [292.12421875, 250.32421875]
    return joint_offset_list, gripper_offset_list