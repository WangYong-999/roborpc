from typing import Union, List, Dict

from roborpc.robots.robot_base import RobotBase


class SimRobotInterface(RobotBase):
    def __init__(self, robot):
        self.robot = robot

    def connect_now(self):
        self.robot.connect_now()

    def disconnect_now(self):
        pass

    def get_robot_ids(self) -> List[str]:
        return self.robot.get_robot_ids()

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robot.set_ee_pose(action, action_space, blocking)

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        self.robot.set_joints(action, action_space, blocking)

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robot.set_gripper(action, action_space, blocking)

    def get_robot_state(self) -> Dict[str, List[float]]:
        return self.robot.get_robot_state()

    def get_dofs(self) -> Dict[str, int]:
        return self.robot.get_dofs()

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_joint_positions()

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_gripper_position()

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_joint_velocities()

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_ee_pose()

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        return self.robot.get_camera_intrinsics()

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        return self.robot.get_camera_extrinsics()

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        return self.robot.read_camera()


