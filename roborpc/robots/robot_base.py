from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union

import numpy as np


class RobotBase(ABC):
    """
    The base class for all robots.
    """

    @abstractmethod
    def connect_now(self):
        """
        Connect to the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect_now(self):
        """
        Disconnect from the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_robot_ids(self) -> List[str]:
        """
        Get the IDs of the robots.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_robot_state(self):
        """
        Reset the state of the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        """
        Set the state of the robot.
        :param state: The state to be set.
        :param blocking: Whether to block.
        """
        raise NotImplementedError

    @abstractmethod
    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "cartesian_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        """
        Update the pose of the robot.
        :param action: The action to be executed.
        :param action_space: The action space of the robot, either "cartesian_position" or "cartesian_velocity".
        :param blocking: Whether to block.
        """
        raise NotImplementedError

    @abstractmethod
    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, Dict[str, str]] = "joint_position",
                   blocking: Union[bool, Dict[str, bool]] = False):
        """
        Update the joints of the robot.
        :param action: The action to be executed.
        :param action_space: The action space of the robot, either "joint_position" or "joint_velocity".
        :param blocking: Whether to block.
        """
        raise NotImplementedError

    @abstractmethod
    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "gripper_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        """
        Update the gripper of the robot.
        :param action: The action to be executed.
        :param action_space: The action space of the robot, either "gripper_position" or "gripper_velocity".
        :param blocking: Whether to block.
        """
        raise NotImplementedError

    @abstractmethod
    def get_robot_state(self) -> Dict[str, Dict[str, List[float]]]:
        """Get the state of the robot.

        Returns:
            A dictionary containing the state of the robot, including "pose", "joints", "gripper_position", "gripper_velocity", "cartesian_velocity", "joint_velocity".
        """
        raise NotImplementedError

    @abstractmethod
    def get_dofs(self) -> Union[int, Dict[str, int]]:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get the joint positions of the robot."""
        raise NotImplementedError

    @abstractmethod
    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get the gripper position of the robot."""
        raise NotImplementedError

    @abstractmethod
    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get the joint velocity of the robot."""
        raise NotImplementedError

    @abstractmethod
    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get the pose (x,y,z,r,p,y) of the robot."""
        raise NotImplementedError
