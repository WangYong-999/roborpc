from typing import Dict, List, Union

import zerorpc
from robot_base import RobotBase
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config
from thirty_party.realman.robotic_arm import *


class MultiRealManRpc(RobotBase):

    def __init__(self, server_ip_address: str, rpc_port: str):
        super().__init__()
        self.server_ip_address = server_ip_address
        self.rpc_port = rpc_port
        self.robots = None

    def connect(self):
        self.robots = zerorpc.Client(heartbeat=20)
        self.robots.connect("tcp://" + self.server_ip_address + ":" + self.rpc_port)
        self.robots.connect()

    def disconnect(self):
        self.robots.disconnect()

    def get_robot_ids(self) -> List[str]:
        return self.robots.get_robot_ids()

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robots.set_ee_pose(action, action_space, blocking)

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        self.robots.set_joints(action, action_space, blocking)

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robots.set_gripper(action, action_space, blocking)

    def get_robot_state(self) -> Dict[str, List[float]]:
        return self.robots.get_robot_state()

    def get_dofs(self) -> Dict[str, int]:
        return self.robots.get_dofs()

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robots.get_joint_positions()

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robots.get_gripper_position()

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robots.get_joint_velocities()

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robots.get_ee_pose()


class ComposedMultiRealManRpc(RobotBase):
    def __init__(self):
        self.robot_config = config["roborpc"]["robots"]["realman"]
        self.robots = None

    def connect(self):
        all_robot_ids = self.robot_config["robot_ids"]
        self.robots = {}
        for robot_id in all_robot_ids:
            ip_address = self.robot_config[robot_id]["server_ip_address"]
            rpc_port = self.robot_config[robot_id]["sever_rpc_port"]
            self.robots[robot_id] = MultiRealManRpc(ip_address, rpc_port)
            self.robots[robot_id].connect()
            logger.success(f"RealMan Robot {robot_id} Connect Success!")

    def disconnect(self):
        for robot_id, robot in self.robots:
            robot.disconnect()
            logger.info(f"RealMan Robot {robot_id} Disconnect Success!")

    def get_robot_ids(self) -> List[str]:
        return self.robot_config["robot_ids"]

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "cartesian_position", blocking: Union[bool, List[bool]] = False):
        for robot_id, robot in self.robots.items():
            robot.set_ee_pose(action, action_space, blocking)
            
    def set_joints(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        for robot_id, robot in self.robots.items():
            robot.set_joints(action, action_space, blocking)

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "gripper_position", blocking: Union[bool, List[bool]] = False):
        for robot_id, robot in self.robots.items():
            robot.set_gripper(action, action_space, blocking)

    def get_robot_state(self) -> Dict[str, List[float]]:
        robot_state = {}
        for robot_id, robot in self.robots.items():
            robot_state[robot_id] = robot.get_robot_state()
        return robot_state

    def get_dofs(self) -> Dict[str, int]:
        dofs = {}
        for robot_id, robot in self.robots.items():
            dofs = robot.get_dofs()
            dofs[robot_id] = dofs
        return dofs

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        joint_positions = {}
        for robot_id, robot in self.robots.items():
            joint_positions[robot_id] = robot.get_joint_positions()
        return joint_positions

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        gripper_positions = {}
        for robot_id, robot in self.robots.items():
            gripper_positions[robot_id] = robot.get_gripper_position()
        return gripper_positions

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        joint_velocities = {}
        for robot_id, robot in self.robots.items():
            joint_velocities[robot_id] = robot.get_joint_velocities()
        return joint_velocities

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        ee_poses = {}
        for robot_id, robot in self.robots.items():
            ee_poses[robot_id] = robot.get_ee_pose()
        return ee_poses

