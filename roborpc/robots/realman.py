from typing import Dict, List, Union, Optional

import numpy as np
import transforms3d.euler
import argparse

from robot_base import RobotBase
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config

# new version of realman API
from thirty_party.realman.robotic_arm import *

# old version of realman API
from thirty_party.realman.realman_driver import DriverRealman


class RealMan(RobotBase):

    def __init__(self, robot_id: str, ip_address: str):
        super().__init__()
        self.robot_id = robot_id
        self.ip_address = ip_address
        self.robot: Optional[DriverRealman] = None
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None

    def connect(self):
        self.robot = DriverRealman()
        self.robot_arm_dof = config["roborpc"]["robots"]["realman"][self.robot_id]["robot_arm_dof"]
        self.robot_gripper_dof = config["roborpc"]["robots"]["realman"][self.robot_id]["robot_gripper_dof"]
        self.last_arm_state = [0.0] * self.robot_arm_dof

        logger.info("Connect to RealMan Robot.")

    def disconnect(self):
        pass

    def get_robot_ids(self) -> List[str]:
        pass

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "cartesian_position", blocking: Union[bool, List[bool]] = False):
        self.robot.move_cartesian_pose_trajectory(np.array([action]))

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        self.robot.move_joints_radian_trajectory(np.array([action]))

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "gripper_position", blocking: Union[bool, List[bool]] = False):
        self.robot.set_gripper_opening(action[0])

    def get_robot_state(self) -> Dict[str, List[float]]:
        robot_state = {"joint_position": self.get_joint_positions(), "gripper_position": self.get_gripper_position(),
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
    def __init__(self, args: argparse.Namespace = None):
        self.action: Union[List[float], Dict[str, List[float]]]
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
        self.robot_ids = self.robot_config["robot_ids"][0]

    def connect(self):
        for robot_id in self.robot_ids:
            ip_address = self.robot_config[robot_id]["ip_address"]
            self.robots[robot_id] = RealMan(robot_id, ip_address)
            self.robots[robot_id].connect()
            self.threads[robot_id] = threading.Thread(target=self.sync_robot_state, args=(robot_id,), daemon=True).start()
            logger.success(f"RealMan Robot {robot_id} Connect Success!")

    def disconnect(self):
        for robot_id, robot in self.robots:
            robot.disconnect()
            self.threads[robot_id].close()
            logger.info(f"RealMan Robot {robot_id} Disconnect Success!")

    def sync_robot_state(self, robot_id: str):
        while True:
            try:
                if self.action_space.get(robot_id) == "joint_position":
                    self.robots[robot_id].set_joints(self.action[robot_id], self.action_space[robot_id], self.blocking[robot_id])
                elif self.action_space.get(robot_id) == "gripper_position":
                    self.robots[robot_id].set_gripper(self.action[robot_id], self.action_space[robot_id], self.blocking[robot_id])
                elif self.action_space.get(robot_id) == "cartesian_position":
                    self.robots[robot_id].set_ee_pose(self.action[robot_id], self.action_space[robot_id], self.blocking[robot_id])
                if self.get_robot_state_flag:
                    self.get_robot_state_flag = False
                    self.robot_state[robot_id] = self.robots[robot_id].get_robot_state()
                if self.get_joint_positions_flag:
                    self.get_joint_positions_flag = False
                    self.joint_positions[robot_id] = self.robots[robot_id].get_joint_positions()
                if self.get_gripper_positions_flag:
                    self.get_gripper_positions_flag = False
                    self.gripper_positions[robot_id] = self.robots[robot_id].get_gripper_position()
                if self.get_joint_velocities_flag:
                    self.get_joint_velocities_flag = False
                    self.joint_velocities[robot_id] = self.robots[robot_id].get_joint_velocities()
                if self.get_ee_pose_flag:
                    self.get_ee_pose_flag = False
                    self.ee_poses[robot_id] = self.robots[robot_id].get_ee_pose()
            except Exception as e:
                logger.error(f"Sync Robot State Error: {e}")

    def get_robot_ids(self) -> List[str]:
        return self.robot_ids

    def set_ee_pose_gripper(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, Dict[str, str]] = "cartesian_position_gripper_position", blocking: Union[bool, Dict[str, bool]] = False):
        self.action = action
        self.action_space = action_space
        self.blocking = blocking

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, Dict[str, str]] = "cartesian_position", blocking: Union[bool, Dict[str, bool]] = False):
        self.action = action
        self.action_space = action_space
        self.blocking = blocking

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, Dict[str, str]] = "joint_position", blocking: Union[bool, Dict[str, bool]] = False):
        self.action = action
        self.action_space = action_space
        self.blocking = blocking

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, Dict[str, str]] = "gripper_position", blocking: Union[bool, Dict[str, bool]] = False):
        self.action = action
        self.action_space = action_space
        self.blocking = blocking

    def get_robot_state(self) -> Dict[str, List[float]]:
        self.get_robot_state_flag = True
        return self.robot_state

    def get_dofs(self) -> Union[int, Dict[str, int]]:
        dofs = {}
        for robot_id, robot in self.robots.items():
            dofs = robot.get_dofs()
            dofs[robot_id] = dofs
        return dofs

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        self.get_joint_positions_flag = True
        return self.joint_positions

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        self.get_gripper_positions_flag = True
        return self.gripper_positions

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        self.get_joint_velocities_flag = True
        return self.joint_velocities

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        self.get_ee_pose_flag = True
        return self.ee_poses


if __name__ == '__main__':
    import zerorpc
    # multi_realman = MultiRealMan()
    # multi_realman.connect()
    # print(multi_realman.get_robot_ids())
    # print(multi_realman.get_robot_state())
    #
    # multi_realman.set_joints([0.10136872295583066, 0.059864793343405505, -0.14184290830957919, -1.8463838156848014,
    #                           0.01965240737745615, -0.2019695010407838, 0.3374869513188684])

    multi_realman = MultiRealMan()
    s = zerorpc.Server(multi_realman)
    robot_id = multi_realman.robot_ids[0]
    rpc_port = multi_realman.robot_config[robot_id]['rpc_port']
    logger.info(f"RPC Server Start on {rpc_port}")
    s.bind(f"tcp://0.0.0.0:{rpc_port}")
    s.run()
