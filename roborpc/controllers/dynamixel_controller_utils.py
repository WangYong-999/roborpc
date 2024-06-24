import time
from typing import Sequence, Optional, Tuple

import tqdm
import numpy as np
from thirty_party.dynamixel.driver import DynamixelDriver
from roborpc.common.logger_loader import logger


def get_joint_offsets(
        joint_ids: Sequence[int],
        joint_signs: Optional[Sequence[int]] = None,
        port: str = "/dev/ttyUSB0",
        gripper_config: Optional[Tuple[int, float, float]] = None,
        start_joints: Optional[np.ndarray] = None,
        baudrate: int = 1000000,
):
    logger.info("Getting joint offsets, please adjust dynamixels robot to starting positions within 3s.")
    for i in tqdm.tqdm(range(3)):
        time.sleep(1)
    driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
    joint_offset_list = []
    gripper_offset_list = []
    num_robot_joints = len(joint_ids)

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = joint_signs[index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = start_joints[index]
        return np.abs(joint_i - start_i)

    for _ in range(10):
        driver.get_joints()  # warmup

    for _ in range(1):
        best_offsets = []
        curr_joints = driver.get_joints()
        for i in range(num_robot_joints):
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
        if gripper_config:
            gripper_offset_list = [int(np.rad2deg(driver.get_joints()[-1]) - 0.2),
                                   int(np.rad2deg(driver.get_joints()[-1]) - 42)]

    driver.close()
    del driver
    logger.success(f"Joint_offset_list: {joint_offset_list}, gripper_offset_list: {gripper_offset_list},"
                   f" Save the joint_offset_list and gripper_offset_list to the config file")
    return joint_offset_list, gripper_offset_list
