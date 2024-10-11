import ast
import json
import math
import time

from transforms3d.derivations.angle_axes import point

from robotic_arm import *
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

    @staticmethod
    def _std_m_to_rm_m(
            value: Union[float, Iterable[float]]) -> Union[int, List[int]]:
        """Convert standard unit (m) to Realman unit (0.001 mm).

        Args:
            value: Length(s) in unit meter.

        Returns:
            Length(s) in unit micron (a.k.a., micrometer, 1e-6 meter).
        """
        if isinstance(value, Iterable):
            return [int(v * 1.e6) for v in value]
        else:
            return int(value * 1.e6)

    @staticmethod
    def _rm_m_to_std_m(
            value: Union[int, Iterable[int]]) -> Union[float, List[float]]:
        """Convert Realman unit (0.001 mm) to standard unit (m).

        Args:
            value: Length(s) in unit micron (a.k.a., micrometer, 1e-6 meter).

        Returns:
            Length(s) in unit meter.
        """
        if isinstance(value, Iterable):
            return [float(v / 1.e6) for v in value]
        else:
            return float(value / 1.e6)

    @staticmethod
    def _std_rad_to_rm_rad(
            value: Union[float, Iterable[float]]) -> Union[int, List[int]]:
        """Convert standard unit (radian) to Realman unit (0.001 radian).

        Args:
            value: Angle(s) in unit radian.

        Returns:
            Angle(s) in unit 0.001 radian.
        """
        if isinstance(value, Iterable):
            return [int(v * 1.e3) for v in value]
        else:
            return int(value * 1.e3)

    @staticmethod
    def _rm_rad_to_std_rad(
            value: Union[int, Iterable[int]]) -> Union[float, List[float]]:
        """Convert Realman unit (0.001 radian) to standard unit (radian).

        Args:
            value: Angle(s) in unit 0.001 radian.

        Returns:
            Angle(s) in unit radian.
        """
        if isinstance(value, Iterable):
            return [float(v / 1.e3) for v in value]
        else:
            return float(value / 1.e3)

    @staticmethod
    def _std_rad_to_rm_deg(
            value: Union[float, Iterable[float]]) -> Union[int, List[int]]:
        """Convert standard unit (radian) to Realman unit (0.001 degree).

        Args:
            value: Angle(s) in unit radian.

        Returns:
            Angle(s) in unit 0.001 degree.
        """
        if isinstance(value, Iterable):
            return [int(math.degrees(v) * 1e3) for v in value]
        else:
            return int(math.degrees(value) * 1e3)

    @staticmethod
    def _rm_deg_to_std_rad(
            value: Union[int, Iterable[int]]) -> Union[float, List[float]]:
        """Convert Realman unit (0.001 degree) to standard unit (radian).

        Args:
            value: Angle(s) in unit 0.001 degree.

        Returns:
            Angle(s) in unit radian.
        """
        if isinstance(value, Iterable):
            return [math.radians(v / 1.e3) for v in value]
        else:
            return math.radians(value / 1.e3)

    # Query Services
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
        print(f"tag:{tag}")
        print(f"state:{state}")
        return 0
        # data = json.dumps({'command': 'get_gripper_state'})
        # data += '\r\n'
        # result = self._send_msg_with_retry(data)
        # print(result)
        # return result['gripper_state']['position'] / 1000.0

    def get_joints_radian(self) -> np.ndarray:
        """Return the joint angles in radian.

        Returns:
            Of shape (n_joints=7,).
        """
        ret = self.robot.Get_Current_Arm_State(retry=1)
        return np.array(ret[1])

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
            joints_in_rm_degree = self._std_rad_to_rm_deg(joints)
            data = json.dumps({
                'command': 'movej_canfd',
                'joint': joints_in_rm_degree,
                'follow': "true"
            })
            data += '\r\n'
            if 'trajectory_state' not in self._send_msg_with_retry(data):
                return False
        return True

    def reach_cartesian_pose(self, pose: np.ndarray) -> Any:
        """Move the arms.

        Args:
            pose: Pose in RotationForm.QUATERNION, of shape (7,).

        Returns:
            The trajectory_state from the response.
        """
        _pose = pose.copy()
        # m to 0.001mm
        _pose[:3] = self._std_m_to_rm_m(pose[:3])
        # rad to 0.001rad
        _pose[3:] = self._std_rad_to_rm_rad(pose[3:])
        data = json.dumps({
            'command': 'movel',
            'pose': _pose,
            'v': self.velocity,
            'r': 0
        })
        data += '\r\n'
        return self._send_msg_with_retry(data)['trajectory_state']

    def reach_joints_radian(self, joints: np.ndarray) -> Any:
        """Move the joints to the given angles.

        Args:
            joints: In radian, of shape (n_joints=7,).

        Returns:
            The trajectory_state from the response.
        """
        # rad to 0.001degree
        _joints = self._std_rad_to_rm_deg(joints)
        data = json.dumps({
            'command': 'movej',
            'joint': _joints,
            'v': self.velocity,
            'r': 0
        })
        data += '\r\n'
        return self._send_msg_with_retry(data)['trajectory_state']

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
        speed = 500
        data = json.dumps({
            'command': 'set_gripper_pick_on',
            'speed': speed,
            'force': force
        })
        data += '\r\n'
        self._send_msg(data, with_ret=True)
        time.sleep(self.gripper_time)
        return True

    def set_gripper_opening(self, position: float) -> bool:
        """Set the gripper's opening amount to a given value.

        Args:
            position: The gripper opening amount to be set, in a range of
                [1.0 (fully closed), 0.0 (fully open)].

        Returns:
            Whether the command is sent.
        """
        data = json.dumps({
            'command': 'set_gripper_position',
            'position': int((1.0 - position) * 1000)
        })
        data += '\r\n'
        self._send_msg_with_retry(data, with_ret=False)
        time.sleep(self.gripper_time)
        self.last_gripper_position = position
        return True

    # ----
    # Lift Control Services
    def reach_lift_height(self, height: float) -> Any:
        """Move the lift to a given height.

        Args:
            height: 0 for stopping; other values for reaching to that height.

        Returns:
            Whether the command is sent.
        """
        if height == 0:
            return self.stop_lift()
        # _height = max(min(height, MAX_LIFT), MIN_LIFT)
        data = json.dumps({
            'command': 'set_lift_height',
            'height': int(height * 1e3),
            'speed': 80
        })
        data += '\r\n'
        return self._send_msg_with_retry(data)['trajectory_state']

    def stop_lift(self) -> Any:
        """Stop the lift.

        Returns:
            Whether the command is sent.
        """
        data = json.dumps({'command': 'set_lift_speed', 'speed': 0})
        data += '\r\n'
        return self._send_msg_with_retry(data)['set_state']


if __name__ == '__main__':
    realman = DriverRealman()
    print(realman.get_joints_radian().tolist())
    print(realman.get_gripper_opening())

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
