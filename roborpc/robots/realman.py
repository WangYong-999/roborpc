import threading
from typing import Dict, List, Union, Optional

import numpy as np
import asyncio
import transforms3d.euler
import argparse

from roborpc.robots.robot_base import RobotBase
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config

# new version of realman API
from thirty_party.realman.robotic_arm import *

# old version of realman API
from thirty_party.realman.realman_driver import DriverRealman


class RealMan(RobotBase):

    def __init__(self, robot_id: str, ip_address: str):
        super().__init__()
        self._joints_target = None
        self._gripper_target = None
        self.robot_id = robot_id
        self.ip_address = ip_address
        self.robot: Optional[DriverRealman] = None
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None

    def connect_now(self):
        self.robot = DriverRealman()
        self.robot_arm_dof = config["roborpc"]["robots"]["realman"][self.robot_id]["robot_arm_dof"]
        self.robot_gripper_dof = config["roborpc"]["robots"]["realman"][self.robot_id]["robot_gripper_dof"]
        self.last_arm_state = [0.0] * self.robot_arm_dof
        logger.info("Connect to RealMan Robot.")

    def disconnect_now(self):
        pass

    def get_robot_ids(self) -> List[str]:
        pass

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        for action_space, action in state.items():
            if action_space == "joint_position":
                start_time = time.time()
                self.robot.move_joints_radian_trajectory(np.array(action))
                # logger.info(f"Move joints to {action} in {time.time() - start_time} seconds.")
            # TODO: gripper bug
            elif action_space == "gripper_position" and action[0] < 0.9:
                self.robot.set_gripper_opening(action[0])
            elif action_space == "cartesian_position":
                self.robot.move_cartesian_pose_trajectory(np.array([action]))

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robot.move_cartesian_pose_trajectory(np.array([action]))

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position",
                   blocking: Union[bool, List[bool]] = False):
        self.robot.move_joints_radian_trajectory(np.array([action]))

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robot.set_gripper_opening(action[0])

    def get_robot_state(self) -> Dict[str, List[float]]:
        robot_state = {"joint_position": self.get_joint_positions(),
                       "gripper_position": self.get_gripper_position(),
                       "ee_pose": self.get_ee_pose()}
        return robot_state

    def get_dofs(self) -> Union[int, Dict[str, int]]:
        return self.robot_arm_dof

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return list(self.robot.get_joints_radian())

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        return [self.robot.get_gripper_opening()]

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        return list(self.robot.get_end_effector_pose())


class MultiRealMan(RobotBase):
    def __init__(self):
        self.ee_action: Union[List[float], Dict[str, List[float]]]
        self.joints_action: Union[List[float], Dict[str, List[float]]]
        self.gripper_action: Union[List[float], Dict[str, List[float]]]
        self.action_space: Union[str, Dict[str, str]] = "joint_position"
        self.blocking: Union[bool, Dict[str, bool]] = False
        self.robot_config = config["roborpc"]["robots"]["realman"]
        self.robots = {}
        self.threads = {}
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
            ip_address = self.robot_config[robot_id]["ip_address"]
            self.robots[robot_id] = RealMan(robot_id, ip_address)
            self.robots[robot_id].connect_now()
            logger.success(f"RealMan Robot {robot_id} Connect Success!")

    def disconnect_now(self):
        for robot_id, robot in self.robots:
            robot.disconnect_now()
            self.threads[robot_id].close()
            logger.info(f"RealMan Robot {robot_id} Disconnect Success!")

    def get_robot_ids(self) -> List[str]:
        return self.robot_ids

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space=None,
                    blocking=None):
        if blocking is None:
            blocking = {"realman_1": False}
        if action_space is None:
            action_space = {"realman_1": "cartesian_position"}
        self.loop.run_until_complete(asyncio.gather(*[robot.set_ee_pose(action[robot_id], action_space[robot_id],
                                                                        blocking[robot_id])
                                                      for robot_id, robot in self.robots.items()]))

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space=None,
                   blocking=None):
        if blocking is None:
            blocking = {"realman_1": False}
        if action_space is None:
            action_space = {"realman_1": "joint_position"}
        self.loop.run_until_complete(asyncio.gather(*[robot.set_joints(action[robot_id], action_space[robot_id],
                                                                       blocking[robot_id])
                                                      for robot_id, robot in self.robots.items()]))

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space=None,
                    blocking=None):
        if blocking is None:
            blocking = {"realman_1": False}
        if action_space is None:
            action_space = {"realman_1": "gripper_position"}
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
