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
                    action_space: Union[str, Dict[str, str]] = "cartesian_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        self.robots.set_ee_pose(action[robot_id], action_space[robot_id], blocking[robot_id])

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, Dict[str, str]] = "joint_position", blocking: Union[bool, Dict[str, bool]] = False):
        self.robots.set_joints(action[robot_id], action_space[robot_id], blocking[robot_id])

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "gripper_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        self.robots.set_gripper(action[robot_id], action_space[robot_id], blocking[robot_id])

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
        self.robot_ids_server_ips = {}
        self.robot_config = config["roborpc"]["robots"]["realman"]
        self.multi_robots = {}

    def connect(self):
        server_ips_address = self.robot_config["server_ips_address"]
        sever_rpc_ports = self.robot_config["sever_rpc_ports"]
        for server_ip_address, sever_rpc_port in zip(server_ips_address, sever_rpc_ports):
            self.multi_robots[server_ip_address] = MultiRealManRpc(server_ip_address, sever_rpc_port)
            self.multi_robots[server_ip_address].connect()
            logger.info(f"RealMan Robot {server_ip_address}:{rpc_port} Connect Success!")
        self.robot_ids_server_ips = self.get_robot_ids_server_ips()

    def disconnect(self):
        for server_ip_address, robot in self.multi_robots.items():
            robot.disconnect()
            logger.info(f"RealMan Robot {server_ip_address} Disconnect Success!")

    def get_robot_ids_server_ips(self) -> Dict[str, str]:
        robot_ids_server_ips = {}
        for server_ip_address, robot in self.multi_robots.items():
            for robot_ids in robot.get_robot_ids():
                for robot_id in robot_ids:
                    robot_ids_server_ips[robot_id] = server_ip_address
        return robot_ids_server_ips

    def get_robot_ids(self) -> List[str]:
        robot_ids = []
        for server_ip_address, robot in self.multi_robots.items():
            robot_ids.extend(robot.get_robot_ids())
        return robot_ids

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, Dict[str, str]] = "cartesian_position", blocking: Union[bool, Dict[str, bool]] = False):
        for server_ip_address, robot in self.multi_robots.items():
            robot.set_ee_pose(action[robot_id], action_space[robot_id], blocking[robot_id])

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, Dict[str, str]] = "joint_position", blocking: Union[bool, Dict[str, bool]] = False):
        for robot_id in self.robot_ids:
            robot.set_joints(action[robot_id], action_space[robot_id], blocking[robot_id])

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, Dict[str, str]] = "gripper_position", blocking: Union[bool, Dict[str, bool]] = False):
        for robot_id in self.robot_ids:
            multi_robots.set_gripper(action[robot_id], action_space[robot_id], blocking[robot_id])

    def get_robot_state(self) -> Dict[str, List[float]]:
        robot_states = {}
        for robot_id in self.robot_ids:
            robot_states[robot_id] = multi_robots.get_robot_state()
        return robot_states

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

