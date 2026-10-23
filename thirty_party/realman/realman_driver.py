import ast
import json
import math
import time
from html.parser import piclose

from transforms3d.derivations.angle_axes import point

from thirty_party.realman.robotic_arm import *
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from roborpc.common.logger_loader import logger


class DriverRealman:
    def __init__(self):
        """The initialization of Realman arm driver.
        """

        self._joints_target = None
        arm_velocity = 50  # Percentage of the Realman max speed.
        device_ip = "192.168.1.18"
        self.robot = Arm(ECO65, device_ip)
        self.robot.Set_Gripper_Release(500)
        self.reach_joints_radian([0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0])


    def get_end_effector_pose(self) -> np.ndarray:
        """Return the effector pose in RotationForm.QUATERNION.

        Returns:
            Of shape (7,). That is Position (x, y, z) + Rotation (x, y, z, w).
        """
        return np.zeros(7)

    def get_gripper_opening(self) -> float:
        """Return the gripper's opening amount."""
        # raise NotImplementedError('get_gripper not implemented')
        tag, state =  self.robot.Get_Gripper_State()
        # print(f"state.actpos:{state.actpos}\n,"
        #       f"state.enable_state:{state.enable_state}\n,"
        #       f"state.status:{state.status}\n,"
        #       f"state.error:{state.error}\n,"
        #       f"state.mode:{state.mode}\n,"
        #       f"sate.current_force:{state.current_force}\n,"
        #       f"state.temperature:{state.temperature}")
        return 1- state.temperature / 1000

    def get_joints_radian(self) -> np.ndarray:
        """Return the joint angles in radian.

        Returns:
            Of shape (n_joints=7,).
        """
        ret = self.robot.Get_Current_Arm_State(retry=1)
        joints_angle = np.deg2rad(ret[1])
        return joints_angle

    # ----
    # Arm Control Services
    def move_cartesian_pose_trajectory(self, waypoints: np.ndarray) -> bool:
        """Move arm following a list of Cartesian space poses.

        Args:
            waypoints: List of RotationForm.QUATERNION, of shape [n_steps, 7].

        Returns:
            Whether the command is sent.
        """
        raise NotImplementedError(
            'move_cartesian_pose_trajectory not implemented.')

    def move_joints_radian_trajectory(self, wayjoints: np.ndarray) -> bool:
        """Move arm following a list of joint angles.

        Args:
            wayjoints: Joint angles in radian, of shape [n_steps, n_joints=7].

        Returns:
            Whether the command is sent.
        """
        for joints in wayjoints:
            # rad to 0.001degree
            result = np.rad2deg(joints).tolist()
            self.robot.Movej_CANFD(result, False)
            # time.sleep(0.005)
        return True

    def reach_cartesian_pose(self, pose: np.ndarray) -> Any:
        """Move the arms.

        Args:
            pose: Pose in RotationForm.QUATERNION, of shape (7,).

        Returns:
            The trajectory_state from the response.
        """
        raise NotImplementedError(
            'reach_cartesian_pose not implemented.')

    def reach_joints_radian(self, joints: np.ndarray) -> Any:
        """Move the joints to the given angles.

        Args:
            joints: In radian, of shape (n_joints=7,).

        Returns:
            The trajectory_state from the response.
        """
        # rad to 0.001degree
        _joints = np.rad2deg(joints).tolist()
        self.robot.Movej_Cmd(_joints, 20, 0, 0, True)
        return True

    def reach_named_pose(self, name: str) -> Any:
        """Move the arm joints to the (predefined) named pose.

        Args:
            name: Only 'Zero' and 'Home' are supported.

        Returns:
            The trajectory_state from the response.

        Raises:
            NotImplementedError: if the name can not be recognized.
        """
        if name == 'Zero':
            return self.reach_joints_radian(
                np.array(self.cfg['named_pose']['zero_pose']))
        elif name == 'Home':
            return self.reach_joints_radian(
                np.array(self.cfg['named_pose']['home_pose']))
        else:
            raise NotImplementedError(f'Unknown named pose {name}, '
                                      'available: Zero, Home.')

    # ----
    # Gripper Control Services
    def set_gripper_force(self, force: float = 200) -> bool:
        """Set the gripper's force to a pre-defined value.

        Currently, we fix the values: 200 for force and 500 for speed.

        Returns:
            Whether the command is sent.
        """
        return True

    def set_gripper_opening(self, position: float, block: bool = False, timeout: float = 3) -> bool:
        """Set the gripper's opening amount to a given value.

        Args:
            position: The gripper opening amount to be set, in a range of
                [1.0 (fully closed), 0.0 (fully open)].

        Returns:
            Whether the command is sent.
        """
        position = 1 - position
        tag = self.robot.Set_Gripper_Position(int(position*1000), block=block, timeout=timeout)
        return True



if __name__ == '__main__':
    realman = DriverRealman()
    # print(realman.get_joints_radian().tolist())
    #
    realman.reach_joints_radian([0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0])
    print(realman.get_gripper_opening())
    realman.set_gripper_opening(0, block=True)
    print(realman.get_gripper_opening())

    # while True:
    #     try:
    #         result =realman.reach_joints_radian(np.array([-0.2530378349541379, -0.07663740745507101, 0.08124507668033605,
    #                                                       -1.495572636033941, 0.03830997708127553, -1.5692953436381816, -0.17636552091402702]))
    #         print(result)
    #         break
    #     except Exception as e:
    #         print(e)
    #         break
