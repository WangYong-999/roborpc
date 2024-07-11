from typing import Union, List, Dict, Sequence, Optional, Tuple

import numpy as np

from roborpc.robots.robot_base import RobotBase
from thirty_party.dynamixel.driver import DynamixelDriver, DynamixelDriverProtocol


class Dynamixel(RobotBase):
    def __init__(self, robot_id: str,
                 joint_ids: Sequence[int],
                 joint_offsets: Optional[Sequence[float]] = None,
                 joint_signs: Optional[Sequence[int]] = None,
                 port: str = "/dev/ttyUSB0",
                 gripper_config: Optional[Tuple[int, float, float]] = None,
                 start_joints: Optional[np.ndarray] = None,
                 baudrate: int = 1000000,
                 ):
        super().__init__()

        self.robot_id = robot_id
        self._torque_on = False
        self._last_pos = None
        self._alpha = 0.99
        self._joint_ids = joint_ids
        self.joint_offsets = joint_offsets
        self.joint_signs = joint_signs
        self.port = port
        self.start_joints = start_joints
        self._driver: Optional[DynamixelDriverProtocol] = None
        self.gripper_open_close: Optional[Tuple[float, float]]
        self.gripper_config = gripper_config
        self.baudrate = baudrate

        if gripper_config is not None:
            assert joint_offsets is not None
            assert joint_signs is not None
            self.joint_ids = tuple(joint_ids) + (gripper_config[0],)
            self.joint_offsets = tuple(joint_offsets) + (0.0,)
            self.joint_signs = tuple(joint_signs) + (1,)
            self.gripper_open_close = (
                gripper_config[1] * np.pi / 180,
                gripper_config[2] * np.pi / 180,
            )
        else:
            self.gripper_open_close = None

        if self.joint_offsets is None:
            self._joint_offsets = np.zeros(len(self.joint_ids))
        else:
            self._joint_offsets = np.array(self.joint_offsets)

        if self.joint_signs is None:
            self._joint_signs = np.ones(len(self.joint_ids))
            self._joint_ids = self.joint_ids
        else:
            self._joint_signs = np.array(self.joint_signs)
            self._joint_ids = self.joint_ids

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(
            np.abs(self._joint_signs) == 1
        ), f"joint_signs: {self._joint_signs}"

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        print(f"self._joint_ids: {self._joint_ids}")
        self._driver = DynamixelDriver(self._joint_ids, port=self.port, baudrate=self.baudrate)
        self._driver.set_torque_mode(False)

        if self.start_joints is not None:
            new_joint_offsets = []
            current_joints = np.array(self.get_robot_state()["robot_positions"])
            assert current_joints.shape == self.start_joints.shape
            if self.gripper_config is not None:
                current_joints = current_joints[:-1]
                start_joints = self.start_joints[:-1]
            else:
                start_joints = self.start_joints
                current_joints = current_joints
            for c_joint, s_joint, joint_offset in zip(
                    current_joints, start_joints, self._joint_offsets
            ):
                new_joint_offsets.append(
                    np.pi * 2 * np.round((s_joint - c_joint) / (2 * np.pi))
                    + joint_offset
                )
            if self.gripper_config is not None:
                new_joint_offsets.append(self._joint_offsets[-1])
            self._joint_offsets = np.array(new_joint_offsets)
        return True

    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        return True

    def get_robot_ids(self) -> List[str]:
        pass

    def reset_robot_state(self):
        pass

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        pass

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        self._driver.set_joints((np.array(action) + self._joint_offsets).tolist())

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def get_robot_state(self) -> Dict[str, List[float]]:
        pos = (self._driver.get_joints() - self._joint_offsets) * self._joint_signs
        assert len(pos) == len(self._joint_ids)

        if self.gripper_open_close is not None:
            # map pos to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                    self.gripper_open_close[1] - self.gripper_open_close[0]
            )
            g_pos = min(max(0, g_pos), 1)
            pos[-1] = g_pos

        if self._last_pos is None:
            self._last_pos = pos
        else:
            # exponential smoothing
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos
        robot_states = {
            # "cartesian_position": end_effector_world_pose,
            "gripper_position": list(pos[-1:]),
            "joint_position": list(pos[:-1]),
            "robot_positions": list(pos),
            # "joint_velocities": arm_joints_velocities,
            # "joint_torques_computed": arm_joint_torques_computed,
            # "prev_joint_torques_computed": arm_joint_torques_computed,
            # "prev_joint_torques_computed_safened": arm_joint_torques_computed,
            # "motor_torques_measured": arm_joint_torques_computed,
            # "prev_controller_latency_ms": 0.0,
            # "prev_command_successful": True,
        }

        return robot_states

    def get_dofs(self) -> Dict[str, int]:
        pass

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def set_torque_mode(self, mode: bool):
        if mode == self._torque_on:
            return
        self._driver.set_torque_mode(mode)
        self._torque_on = mode
