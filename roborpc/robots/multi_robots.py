import asyncio
from typing import Union, List, Dict

from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.robots.panda import Panda
from roborpc.robots.realman import RealMan
from robot_base import RobotBase


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
        self.loop = asyncio.get_event_loop()
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

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        self.loop.run_until_complete(asyncio.gather(*[robot.set_robot_state(state[robot_id], blocking[robot_id])
                                                      for robot_id, robot in self.robots.items()]))

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "cartesian_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        self.loop.run_until_complete(asyncio.gather(*[robot.set_ee_pose(action[robot_id], action_space[robot_id],
                                                                        blocking[robot_id])
                                                      for robot_id, robot in self.robots.items()]))

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, Dict[str, str]] = "joint_position",
                   blocking: Union[bool, Dict[str, bool]] = False):
        self.loop.run_until_complete(asyncio.gather(*[robot.set_joints(action[robot_id], action_space[robot_id],
                                                                       blocking[robot_id])
                                                      for robot_id, robot in self.robots.items()]))

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "gripper_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        self.loop.run_until_complete(asyncio.gather(*[robot.set_gripper(action[robot_id], action_space[robot_id],
                                                                        blocking[robot_id])
                                                      for robot_id, robot in self.robots.items()]))

    def get_robot_state(self) -> Dict[str, List[float]]:
        robot_state = {}
        for robot_id, robot in self.robots.items():
            robot_state[robot_id] = asyncio.ensure_future(robot.get_robot_state())
        self.loop.run_until_complete(asyncio.gather(*robot_state.values()))
        for robot_id, state in robot_state.items():
            self.robot_state[robot_id] = state.result()
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
            joint_positions[robot_id] = asyncio.ensure_future(robot.get_joint_positions())
        self.loop.run_until_complete(asyncio.gather(*joint_positions.values()))
        for robot_id, positions in joint_positions.items():
            self.joint_positions[robot_id] = positions.result()
        return self.joint_positions

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        gripper_positions = {}
        for robot_id, robot in self.robots.items():
            gripper_positions[robot_id] = asyncio.ensure_future(robot.get_gripper_position())
        self.loop.run_until_complete(asyncio.gather(*gripper_positions.values()))
        for robot_id, positions in gripper_positions.items():
            self.gripper_positions[robot_id] = positions.result()
        return self.gripper_positions

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        joint_velocities = {}
        for robot_id, robot in self.robots.items():
            joint_velocities[robot_id] = asyncio.ensure_future(robot.get_joint_velocities())
        self.loop.run_until_complete(asyncio.gather(*joint_velocities.values()))
        for robot_id, velocities in joint_velocities.items():
            self.joint_velocities[robot_id] = velocities.result()
        return self.joint_velocities

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        ee_poses = {}
        for robot_id, robot in self.robots.items():
            ee_poses[robot_id] = asyncio.ensure_future(robot.get_ee_pose())
        self.loop.run_until_complete(asyncio.gather(*ee_poses.values()))
        for robot_id, pose in ee_poses.items():
            self.ee_poses[robot_id] = pose.result()
        return self.ee_poses




