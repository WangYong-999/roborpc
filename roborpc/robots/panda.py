from typing import Union, List, Dict

from roborpc.robots.robot_base import RobotBase


class Panda(RobotBase):

    def __init__(self, robot_id: str, ip_address: str):
        super().__init__()

    def connect_now(self):
        pass

    def disconnect_now(self):
        pass

    def get_robot_ids(self) -> List[str]:
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


