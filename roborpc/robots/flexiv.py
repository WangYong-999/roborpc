import threading
import time
from typing import Dict, List, Union, Optional

import numpy as np
import asyncio
import transforms3d.euler
import argparse

from roborpc.robots.robot_base import RobotBase
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config

# Flexiv API
from airexo.device import flexivrdk
from airexo.helpers.rotation import quat_angle
from airexo.helpers.logger import ColoredLogger


class Flexiv(RobotBase):

    def __init__(self, robot_id: str, ip_address: str):
        super().__init__()
        self._joints_target = None
        self._gripper_target = None
        self.robot_id = robot_id
        self.ip_address = ip_address
        self.robot: Optional[flexivrdk.Robot] = None
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None

        # flexiv
        self.mode = flexivrdk.Mode
        self.current_mode = None
        self.impedance = False
        self.threshold_joint = 0.001
        self.threshold_xyz = 0.001
        self.threshold_rot = 0.001
        self.waiting_gap = 0.05
        self.timeout = 10
        self.impedance_joint_stiffness = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        self.impedance_joint_damping_ratio = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],

        self.max_vel_safe = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        self.max_acc_safe = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.max_vel_rt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.max_acc_rt = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    def connect_now(self):
        self.robot = flexivrdk.Robot(self.ip_address)
        if self.robot.fault():
            if not self.robot.ClearFault():
                raise RuntimeError("Cannot clear faults on the robot.")
        self.robot.Enable()
        while not self.robot.operational():
            time.sleep(0.5)

        self.robot_arm_dof = config["roborpc"]["robots"]["flexiv"][self.robot_id]["robot_arm_dof"]
        self.robot_gripper_dof = config["roborpc"]["robots"]["flexiv"][self.robot_id]["robot_gripper_dof"]
        self.last_arm_state = [0.0] * self.robot_arm_dof
        logger.info("Connect to Flexiv Robot.")

    def disconnect_now(self):
        self.robot.Stop()

    def get_robot_ids(self) -> List[str]:
        pass

    def reset_robot_state(self):
        pass

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        for action_space, action in state.items():
            if action_space == "joint_position":
                self.set_joints(action, action_space)
            elif action_space == "joint_impedance":
                self.set_joint_impedance(self.impedance_joint_stiffness,self.impedance_joint_damping_ratio)
                self.set_joints(action, action_space)
            if action_space == "gripper_position" :
                # TODO
                self.set_gripper(action[0], action_space)

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def switch_mode(self, mode):
        mode = getattr(self.mode, mode)
        if mode != self.current_mode:
            self.robot.SwitchMode(mode)
            self.current_mode = mode

    def set_joint_impedance(
            self,
            stiffness,
            damping_ratio
    ) -> None:
        self.impedance = True
        self.switch_mode("NRT_JOINT_IMPEDANCE")
        self.robot.SetJointImpedance(stiffness, damping_ratio)

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position",
                   blocking: Union[bool, List[bool]] = False):
        # self.robot.move_joints_radian_trajectory(np.array([action]))
        if self.impedance:
            self.switch_mode("NRT_JOINT_IMPEDANCE")
            max_vel = self.max_vel_rt
            max_acc = self.max_acc_rt
        else:
            self.switch_mode("NRT_JOINT_POSITION")
            max_vel = self.max_vel_safe
            max_acc = self.max_acc_safe

        # use ours
        target_vel = [0] * self.robot_arm_dof
        target_acc = [0] * self.robot_arm_dof

        self.robot.SendJointPosition(
            action,
            target_vel,
            target_acc,
            max_vel,
            max_acc
        )
        if blocking:
            if self.impedance:
                logger.warning("The blocking parameter is unavailable and thus ignored for impedance control.")
            else:
                self.wait_for_joint_move(action)

    def wait_for_joint_move(self, target_pos):
        start_time = time.time()
        has_warned = False
        while True:
            if np.max(np.abs(target_pos - self.get_joint_positions())) <= self.threshold_joint:
                break
            if time.time() - start_time > self.timeout and not has_warned:
                logger.warning("Timeout for send_joint_pos.")
                has_warned = True

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        # TODO
        pass

    def get_robot_state(self) -> Dict[str, Dict[str, List[float]]]:

        robot_states = self.robot.states()
        robot_state = {"joint_position": robot_states.q,
                       "joint_velocity": robot_states.dq,
                       "gripper_position": self.get_gripper_position(),
                       "cartesian_position": self.get_ee_pose()}
        return robot_state

    def get_dofs(self) -> Union[int, Dict[str, int]]:
        return self.robot_arm_dof

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return list(self.robot.states().q)

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        return [self.robot.get_gripper_opening()]

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        return list(self.robot.get_end_effector_pose())

