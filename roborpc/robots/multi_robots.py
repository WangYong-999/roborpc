import asyncio
import time
from typing import Union, List, Dict

from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.robots.panda import Panda
from roborpc.robots.realman import RealMan
from roborpc.robots.robot_base import RobotBase


class MultiRobots(RobotBase):

    def __init__(self):
        self.ee_action: Union[List[float], Dict[str, List[float]]]
        self.joints_action: Union[List[float], Dict[str, List[float]]]
        self.gripper_action: Union[List[float], Dict[str, List[float]]]
        self.action_space: Union[str, Dict[str, str]] = "joint_position"
        self.blocking: Union[bool, Dict[str, bool]] = False
        self.robot_config = config["roborpc"]["robots"]
        self.robots = {}
        self.robot_state = {}
        self.joint_positions = {}
        self.gripper_positions = {}
        self.joint_velocities = {}
        self.ee_poses = {}
        self.get_robot_state_flag = False
        self.get_joint_positions_flag = False
        self.get_gripper_positions_flag = False
        self.get_joint_velocities_flag = False
        self.get_ee_pose_flag = False
        self.robot_ids = self.robot_config["robot_ids"][0]

    def connect_now(self):
        for robot_id in self.robot_ids:
            if str(robot_id).startswith("realman"):
                ip_address = self.robot_config["realman"][robot_id]["ip_address"]
                self.robots[robot_id] = RealMan(robot_id, ip_address)
            elif str(robot_id).startswith("panda"):
                ip_address = self.robot_config["panda"][robot_id]["ip_address"]
                self.robots[robot_id] = Panda(robot_id, ip_address)
            else:
                logger.error(f"Robot {robot_id} is not supported!")
            self.robots[robot_id].connect_now()
            logger.success(f"Robot {robot_id} Connect Success!")

    def disconnect_now(self):
        for robot_id, robot in self.robots.items():
            robot.disconnect_now()
            logger.info(f"Robot {robot_id} Disconnect Success!")

    def get_robot_ids(self) -> List[str]:
        return self.robot_ids

    def reset_robot_state(self):
        robot_state = {}
        blocking_info = {}
        for robot_id, robot in self.robots.items():
            start_arm_joints = self.robot_config[robot_id]['start_arm_joints']
            start_gripper_position = self.robot_config[robot_id]['start_gripper_position']
            robot_state[robot_id] = {'joint_position': start_arm_joints, 'gripper_position': start_gripper_position}
            blocking_info[robot_id].update({'joint_position': True, 'gripper_position': True})
        self.set_robot_state(robot_state, blocking_info)

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        for robot_id, robot in self.robots.items():
            robot.set_robot_state(state[robot_id], blocking[robot_id])

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "cartesian_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        [robot.set_ee_pose(action[robot_id], action_space[robot_id],
                           blocking[robot_id])
         for robot_id, robot in self.robots.items()]

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, Dict[str, str]] = "joint_position",
                   blocking: Union[bool, Dict[str, bool]] = False):
        [robot.set_joints(action[robot_id], action_space[robot_id],
                          blocking[robot_id])
         for robot_id, robot in self.robots.items()]

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "gripper_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        [robot.set_gripper(action[robot_id], action_space[robot_id],
                           blocking[robot_id])
         for robot_id, robot in self.robots.items()]

    def get_robot_state(self) -> Dict[str, List[float]]:
        robot_state = {}
        for robot_id, robot in self.robots.items():
            robot_state[robot_id] = robot.get_robot_state()
        return self.robot_state

    def get_dofs(self) -> Union[int, Dict[str, int]]:
        dofs = {}
        for robot_id, robot in self.robots.items():
            dofs = robot.get_dofs()
            dofs[robot_id] = dofs
        return dofs

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        joint_positions = {}
        for robot_id, robot in self.robots.items():
            joint_positions[robot_id] = robot.get_joint_positions()
        return self.joint_positions

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        gripper_positions = {}
        for robot_id, robot in self.robots.items():
            gripper_positions[robot_id] = robot.get_gripper_position()
        return self.gripper_positions

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        joint_velocities = {}
        for robot_id, robot in self.robots.items():
            joint_velocities[robot_id] = asyncio.ensure_future(robot.get_joint_velocities())
        return self.joint_velocities

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        ee_poses = {}
        for robot_id, robot in self.robots.items():
            ee_poses[robot_id] = robot.get_ee_pose()
        return self.ee_poses


if __name__ == '__main__':
    import zerorpc
    multi_robots = MultiRobots()
    s = zerorpc.Server(multi_robots)
    rpc_port = multi_robots.robot_config['sever_rpc_ports'][0]
    logger.info(f"RPC Server Start on {rpc_port}")
    s.bind(f"tcp://0.0.0.0:{rpc_port}")
    s.run()



