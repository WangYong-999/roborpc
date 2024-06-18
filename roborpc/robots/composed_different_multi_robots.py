from typing import Union, List, Dict

from roborpc.common.config_loader import config
from robot_base import RobotBase


class ComposedDifferentMultiRobots(RobotBase):

    def __init__(self):
        self.robots = {}
        self.robot_config = config["roborpc"]["robots"]

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_robot_ids(self) -> List[str]:
        pass

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        pass

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def get_robot_state(self) -> Dict[str, List[float]]:
        pass

    def get_dofs(self) -> Union[int, Dict[str, int]]:
        pass

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        pass



