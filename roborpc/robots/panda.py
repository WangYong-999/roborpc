from typing import Union, List, Dict

import numpy as np
import time

from roborpc.cameras.transformations import ur_axis_angle_to_quat, quat_to_euler
from roborpc.robots.robot_base import RobotBase
from roborpc.common.logger_loader import logger
import torch


class Panda(RobotBase):

    def __init__(self, robot_id: str, ip_address: str):
        super().__init__()
        self.robot_id = robot_id
        self.ip_address = ip_address
        self.robot = None
        self.r_inter = None
        self.gripper = None
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None
        self._free_drive = False
        self.gripper_max_width = 0.09

    def connect_now(self):
        from polymetis import GripperInterface, RobotInterface
        try:
            self.robot = RobotInterface(ip_address=self.ip_address)
        except Exception as e:
            logger.error(f"Failed to connect to Panda robot {self.robot_id} at {self.ip_address}: {e}")
        self.gripper = GripperInterface(ip_address="localhost")
        self.gripper.connect(hostname=self.ip_address, port=63352)
        self.robot.go_home()
        self.robot.start_joint_impedance()
        self.gripper.goto(width=self.gripper_max_width, speed=255, force=255)
        time.sleep(1)
        logger.success("Connected to UR robot")

    def disconnect_now(self):
        pass

    def get_robot_ids(self) -> List[str]:
        pass

    def reset_robot_state(self):
        pass

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        for action_space, action in state.items():
            if action_space == "joint_position":
                self.robot.update_desired_joint_positions(torch.tensor(action))
            elif action_space == "gripper_position":
                self.gripper.goto(width=(self.gripper_max_width * (1 - action[0])), speed=1, force=1)
            elif action_space == "cartesian_position":
                pass

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "cartesian_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        pass

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, Dict[str, str]] = "joint_position",
                   blocking: Union[bool, Dict[str, bool]] = False):
        pass

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "gripper_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        pass

    def get_robot_state(self) -> Dict[str, Dict[str, List[float]]]:
        robot_state = {"joint_position": self.get_joint_positions(),
                       "gripper_position": self.get_gripper_position(),
                       "cartesian_position": self.get_ee_pose()}
        return robot_state

    def get_dofs(self) -> Union[int, Dict[str, int]]:
        pass

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_joint_positions().tolist()

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.gripper.get_state() / self.gripper_max_width

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        pos, quat = self.robot.get_ee_pose()
        angle = quat_to_euler(quat.numpy())
        return np.concatenate([pos, angle]).tolist()


