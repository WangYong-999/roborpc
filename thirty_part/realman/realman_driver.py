import ast
import json
import math
import socket
import time
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from roborpc.common.logger_loader import logger


class DriverRealman:
    def __init__(self):
        """The initialization of Realman arm driver.
        """

        arm_velocity = 25  # Percentage of the Realman max speed.
        device_ip = "192.168.1.18"
        device_port = 8080
        gripper_time = 2.0  # in seconds
        retry = 3  # times
        retry_delay = 1.0  # in seconds
        socket_time_out = 2.0  # in seconds

        home_pose = [0.0, -1.9, 0.0, 0.0, 0.0, 1.9,
                     0.0]  # [0.0, -1.5708, 0.0, 0.0, 0.0, 1.5708, 0.0]  # unit: radian, TODO(zhiyuan), update home pose
        zero_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # unit: radian
        tray_pose = [-0.2775929272174835, -0.7669379711151123, -3.6843175888061523,
                     0.8580743074417114, 3.157172203063965, 4.292811393737793,
                     2.383378267288208]


        self.realman_ip = device_ip
        self.realman_port = device_port
        self.time_out = socket_time_out
        self.delay = retry_delay
        self.velocity = arm_velocity
        self.retry = retry
        self.arm_velocity = arm_velocity
        self.home_pose = home_pose
        self.zero_pose = zero_pose
        self.tray_pose = tray_pose
        self.gripper_time = gripper_time
        self.socket: Optional[socket.socket] = None
        self._connect()

        self.last_gripper_position = 1.0
        self.set_gripper_opening(1.0)

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

    def _connect(self) -> None:
        """Establish the connection to the realman controller."""
        if self.socket:
            self.socket.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.time_out)
        self.socket.connect((self.realman_ip, self.realman_port))
        logger.success(f'Connected to Realman robot '
                       f'{self.realman_ip}:{self.realman_port}')

    def _send_msg(self, msg: str, with_ret: bool = True) -> str:
        """Send messages to the Realman driver.

        Args:
            msg: messages to send.
            with_ret: Whether response value needs to return.

        Returns:
            A literal string, in which response values are stored as a dict.

        Raises:
            ConnectionError: The socket is not created or is broken.
        """
        if self.socket is None:
            raise ConnectionError('The socket is not created or is broken.')
        self.socket.sendall(bytes(msg, encoding='utf-8'))
        if not with_ret:
            return '{\'result\': None}'
        data = b''
        btime = time.time()
        while True:
            if time.time() - btime > self.time_out:
                logger.warning(f'Communication in _send_msg timed out with '
                               f'incomplete data: {data!r}.')
                break
            data += self.socket.recv(1024)
            if b'\r\n' in data:
                break
        val = data.decode('utf-8')
        return val

    def _send_msg_with_retry(self,
                             msg: str,
                             with_ret: bool = True) -> Dict[str, Any]:
        """Send messages to the Realman driver with self.retry times attempts.

        Args:
            msg: messages to send.
            with_ret: Whether response value needs to return.

        Returns:
            Responses.

        Raises:
            Communication error with the Realman driver.
        """
        error_info = ''
        for idx in range(self.retry):
            try:
                my_string = self._send_msg(msg, with_ret)
                new_string = my_string.replace('false', 'False').replace('true', 'True')
                return dict(ast.literal_eval(new_string))
            except (SyntaxError, BrokenPipeError, socket.timeout) as e:
                logger.error(f'Failed with {idx + 1} retry(s) and error '
                             f'{type(e)} {e}.')
                error_info = str(e)
                time.sleep(self.delay)
                self._connect()
        raise ConnectionError(f'Communication Error, type: {error_info}.')

    # ----
    # Query Services
    def get_end_effector_pose(self) -> np.ndarray:
        """Return the effector pose in RotationForm.QUATERNION.

        Returns:
            Of shape (7,). That is Position (x, y, z) + Rotation (x, y, z, w).
        """
        data = json.dumps({'command': 'get_current_arm_state'})
        data += '\r\n'
        result = self._send_msg_with_retry(data)
        pose = result['arm_state']['pose']
        # 0.001mm to m
        pose[:3] = self._rm_m_to_std_m(pose[:3])
        # 0.001rad to rad
        pose[3:] = self._rm_rad_to_std_rad(pose[3:])
        return np.array(pose)

    def get_gripper_opening(self) -> float:
        """Return the gripper's opening amount."""
        # raise NotImplementedError('get_gripper not implemented')
        return self.last_gripper_position

    def get_joints_radian(self) -> np.ndarray:
        """Return the joint angles in radian.

        Returns:
            Of shape (n_joints=7,).
        """
        data = json.dumps({'command': 'get_current_arm_state'})
        data += '\r\n'
        result = self._send_msg_with_retry(data)
        pose = result['arm_state']['joint']
        # 0.001degree to rad
        pose = self._rm_deg_to_std_rad(pose)
        return np.array(pose)

    def get_lift_height(self) -> np.ndarray:
        """Return the lift relative height in unit mm.

        The height is relative to a (determined but uncontrollable/unfixed)
        zero height. The driver takes the lift height when the driver launches
        as the zero height.

        Returns:
            Of shape (1,).
        """
        data = json.dumps({'command': 'get_lift_state'})
        data += '\r\n'
        result = self._send_msg_with_retry(data)
        return np.array([result['height'] / 1.e3])

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
                'joint': joints_in_rm_degree
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

    def step_cartesian_pose_increment(self, movement: np.ndarray) -> bool:
        """Move the arms by an incremental value.

        Args:
            movement: of shape (7,).

        Returns:
            The trajectory_state from the response.
        """
        pos_now = self.get_end_effector_pose()
        pos_target = [pos + move for pos, move in zip(pos_now, movement)]
        if self.reach_cartesian_pose(np.array(pos_target)):
            return True
        return False

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
                [0.0 (fully closed), 1.0 (fully open)].

        Returns:
            Whether the command is sent.
        """
        self.last_gripper_position = position
        data = json.dumps({
            'command': 'set_gripper_position',
            'position': int(position * 1000)
        })
        data += '\r\n'
        self._send_msg_with_retry(data, with_ret=False)
        time.sleep(self.gripper_time)
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
    print(realman.get_end_effector_pose())
    print(realman.get_joints_radian().tolist())
    print(realman.get_gripper_opening())
    while True:
        try:
            realman.move_joints_radian_trajectory(np.array([[0.10136872295583066, 0.05989969992844539, -0.14184290830957919,
                                                  -1.846350833211097, 0.01965240737745615, -0.20198695433330377, 0.2375044046113884]]))
            break
        except Exception as e:
            print(e)
            break
