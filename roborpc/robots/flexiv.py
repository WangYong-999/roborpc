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


class Flexiv(RobotBase):

    def __init__(self, robot_id: str, robot_serial_number: str =''):
        super().__init__()
        self.reset_state = True
        self._joints_target = None
        self._gripper_target = None
        self.gripper = None
        self.robot_id = robot_id
        self.robot_serial_number = robot_serial_number
        self.robot: Optional[flexivrdk.Robot] = None
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None

        # flexiv
        self.current_mode = None
        self.impedance = True
        self.threshold_joint = 0.001
        self.threshold_xyz = 0.001
        self.threshold_rot = 0.001
        self.waiting_gap = 0.05
        self.timeout = 10
        self.impedance_joint_stiffness = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0]
        self.impedance_joint_damping_ratio = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]

        # self.max_vel_safe = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        # self.max_acc_safe = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.max_vel_safe = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.max_acc_safe = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # self.max_vel_rt = [2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0]
        # self.max_acc_rt = [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 3.0]

        self.max_vel_rt = [0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0]
        self.max_acc_rt = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0]
        # self.max_vel_rt = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # self.max_acc_rt = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.griper_width = 0.085

    def connect_now(self):
        # Flexiv API
        import flexivrdk
        from roborpc.robots.robotiq_gripper import Robotiq2F85Gripper
        self.mode = flexivrdk.Mode
        if self.robot_serial_number != '':
            self.robot = flexivrdk.Robot(self.robot_serial_number)
        if self.robot.fault():
            if not self.robot.ClearFault():
                raise RuntimeError("Cannot clear faults on the robot.")
        self.robot.Enable()
        while not self.robot.operational():
            print("Waiting for Flexiv Robot to become operational...")
            time.sleep(0.5)

        self.robot_arm_dof = config["roborpc"]["robots"]["flexiv"][self.robot_id]["robot_arm_dof"]
        self.robot_gripper_dof = config["roborpc"]["robots"]["flexiv"][self.robot_id]["robot_gripper_dof"]
        self.last_arm_state = [0.0] * self.robot_arm_dof
        logger.success("Connect to Flexiv Robot.")

        self.gripper = Robotiq2F85Gripper(port="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AB0MLQCE-if00-port0")
        logger.success("Connect to Robotiq Gripper.")


    def disconnect_now(self):
        self.robot.Stop()

    def get_robot_ids(self) -> List[str]:
        pass

    def reset_robot_state(self):
        self.reset_state = True

    def set_robot_state(self, state: Dict[str, List[float]],
                        blocking: Dict[str, bool]):
        for action_space, action in state.items():
            if action_space == "joint_position":
                self.set_joints(action, action_space, blocking["joint_position"])
            if action_space == "gripper_position" :
                self.set_gripper(action , action_space, blocking["gripper_position"])

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def switch_mode(self, mode):
        mode = getattr(self.mode, mode)
        if mode != self.current_mode:
            self.robot.SwitchMode(mode)
            if mode == self.mode.NRT_JOINT_IMPEDANCE:
                self.set_joint_impedance(self.impedance_joint_stiffness, self.impedance_joint_damping_ratio)
            self.current_mode = mode

    def set_joint_impedance(
            self,
            stiffness,
            damping_ratio
    ) -> None:
        self.robot.SetJointImpedance(stiffness, damping_ratio)

    def set_joints(self, action: List[float],
                   action_space: str = "joint_position",
                   blocking: bool = False):
        # self.robot.move_joints_radian_trajectory(np.array([action]))
        if self.reset_state or blocking:
            self.switch_mode("NRT_JOINT_POSITION")
            max_vel = self.max_vel_safe
            max_acc = self.max_acc_safe
        elif not self.reset_state and not blocking and self.impedance:
            self.switch_mode("NRT_JOINT_IMPEDANCE")
            max_vel = self.max_vel_rt
            max_acc = self.max_acc_rt
        else:
            self.switch_mode("NRT_JOINT_POSITION")
            max_vel = self.max_vel_rt
            max_acc = self.max_acc_rt
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
            self.wait_for_joint_move(action)
        self.reset_state = False # for telep start

    def wait_for_joint_move(self, target_pos):
        start_time = time.time()
        has_warned = False
        while True:
            if np.max(np.abs(np.asarray(target_pos) - np.asarray(self.get_joint_positions()))) <= self.threshold_joint:
                break
            if time.time() - start_time > self.timeout and not has_warned:
                logger.warning("Timeout for send_joint_pos.")
                has_warned = True

    def set_gripper(self, action: List[float],
                    action_space: str = "gripper_position",
                    blocking: bool = False):

        if "gripper_position" == action_space:
            self.griper_width = 0.085
            self.gripper.set_width(self.griper_width - action[0] * self.griper_width)
        elif "gripper_velocity" == action_space:
            self.gripper.set_velocity(self.griper_width - action[0] * self.griper_width, action[1] * 255)
        else:
            raise ValueError("Invalid action space for gripper.")


    def get_robot_state(self) -> Dict[str, List[float]]:

        robot_states = self.robot.states()
        robot_state = {"joint_position": robot_states.q,
                       "joint_velocity": robot_states.dq,
                       "gripper_position": self.get_gripper_position(),
                       "cartesian_position": robot_states.tcp_pose,
                       "joint_torque": robot_states.tau,
                       "tcp_ext_wrench": robot_states.ext_wrench_in_tcp,
                       "joint_ext_torque": robot_states.tau_ext,
                       }
        return robot_state

    def get_dofs(self) -> int:
        return self.robot_arm_dof

    def get_joint_positions(self) -> List[float]:
        return list(self.robot.states().q)

    def get_gripper_position(self) -> List[float]:

        return [1.0 - self.gripper.get_width() / self.griper_width ]

    def get_joint_velocities(self) -> List[float]:
        pass

    def get_ee_pose(self) -> List[float]:
        return list(self.robot.states().tcp_pose)

class FlexivV2(RobotBase):

    def __init__(self, robot_id: str, robot_ip_address: str = '', local_ip_address: str = ''):
        super().__init__()
        self.reset_state = True
        self._joints_target = None
        self._gripper_target = None
        self.gripper = None
        self.robot_id = robot_id
        self.robot_ip_address = robot_ip_address
        self.local_ip_address = local_ip_address
        self.robot: Optional[flexivrdk.Robot] = None
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None

        # flexiv
        self.current_mode = None
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

        self.griper_width = 0.085

    def connect_now(self):
        # Flexiv API
        import flexivrdk
        from roborpc.robots.robotiq_gripper import Robotiq2F85Gripper
        self.mode = flexivrdk.Mode
        self.robot_states = flexivrdk.RobotStates()
        if self.robot_ip_address != '' and self.local_ip_address != '':
            self.robot = flexivrdk.Robot(self.robot_ip_address, self.local_ip_address)
        if self.robot.isFault():
            if not self.robot.ClearFault():
                raise RuntimeError("Cannot clear faults on the robot.")
        self.robot.enable()
        while not self.robot.isOperational():
            time.sleep(0.5)

        # Move robot to home pose
        self.robot.setMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.executePrimitive("Home()")

        # Wait for the primitive to finish
        while (self.robot.isBusy()):
            time.sleep(1)

        # Switch to non-real-time joint position control mode
        self.robot.setMode(self.mode.NRT_JOINT_POSITION)

        self.robot_arm_dof = config["roborpc"]["robots"]["flexiv"][self.robot_id]["robot_arm_dof"]
        self.robot_gripper_dof = config["roborpc"]["robots"]["flexiv"][self.robot_id]["robot_gripper_dof"]
        self.last_arm_state = [0.0] * self.robot_arm_dof
        logger.success("Connect to Flexiv Robot.")

        # self.gripper = Robotiq2F85Gripper(port="/dev/ttyUSB0")
        # logger.success("Connect to Robotiq Gripper.")

    def disconnect_now(self):
        self.robot.Stop()

    def get_robot_ids(self) -> List[str]:
        pass

    def reset_robot_state(self):
        self.reset_state = True

    def set_robot_state(self, state: Dict[str, List[float]],
                        blocking: Dict[str, bool]):
        for action_space, action in state.items():
            if action_space == "joint_position":
                self.set_joints(action, action_space)
            if action_space == "gripper_position" :
                self.set_gripper(action , action_space)

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def set_joints(self, action: List[float],
                   action_space: str = "joint_position",
                   blocking: bool = False):
        # self.robot.move_joints_radian_trajectory(np.array([action]))
        if self.reset_state:
            max_vel = self.max_vel_safe
            max_acc = self.max_acc_safe
        else:
            max_vel = self.max_vel_rt
            max_acc = self.max_acc_rt

        # use ours
        target_vel = [0] * self.robot_arm_dof
        target_acc = [0] * self.robot_arm_dof

        self.robot.sendJointPosition(
            action,
            target_vel,
            target_acc,
            max_vel,
            max_acc
        )
        if blocking:
            self.wait_for_joint_move(action)

        self.reset_state = False

    def wait_for_joint_move(self, target_pos):
        start_time = time.time()
        has_warned = False
        while True:
            if np.max(np.abs(target_pos - self.get_joint_positions())) <= self.threshold_joint:
                break
            if time.time() - start_time > self.timeout and not has_warned:
                logger.warning("Timeout for send_joint_pos.")
                has_warned = True

    def set_gripper(self, action: List[float],
                    action_space: str = "gripper_position",
                    blocking: bool = False):

        if "gripper_position" == action_space:
            self.gripper.set_width(self.griper_width - action[0] * self.griper_width)
        elif "gripper_velocity" == action_space:
            self.gripper.set_width(self.griper_width  - action[0] * self.griper_width, action[1] * 255)
        else:
            raise ValueError("Invalid action space for gripper.")


    def get_robot_state(self) -> Dict[str, List[float]]:

        self.robot.getRobotStates(self.robot_states)
        robot_state = {"joint_position": self.robot_states.q,
                       "joint_velocity": self.robot_states.dq,
                       "gripper_position": self.get_gripper_position(),
                       "cartesian_position": self.get_ee_pose(),
                       "joint_torque": self.robot_states.tau,
                       "tcp_ext_wrench": self.robot_states.ext_wrench_in_tcp,
                       "joint_ext_torque": self.robot_states.tau_ext,
                       }
        return robot_state

    def get_dofs(self) -> int:
        return self.robot_arm_dof

    def get_joint_positions(self) -> List[float]:
        self.robot.getRobotStates(self.robot_states)
        return list(self.robot_states.q)

    def get_gripper_position(self) -> List[float]:

        return [1.0 - self.gripper.get_width() / self.griper_width]

    def get_joint_velocities(self) -> List[float]:
        pass

    def get_ee_pose(self) -> List[float]:
        return list(self.robot.get_end_effector_pose())


if  __name__ == '__main__':
    flexiv = Flexiv("flexiv_1", "Rizon4-062206")
    flexiv.connect_now()
    current_joint = flexiv.get_joint_positions()
    print(current_joint)
    # last joint position added 0.1
    current_joint[-1] += 0.1
    flexiv.set_joints(current_joint, blocking=True)
    current_joint[-1] -= 0.1
    flexiv.set_joints(current_joint)

    # flexiv = FlexivV2("flexiv_1", "192.168.2.100", "192.168.2.102")
    # flexiv.connect_now()
    # current_joint = flexiv.get_joint_positions()
    # print(current_joint)
    # # last joint position added 0.1
    # current_joint[-1] += 0.1
    # flexiv.set_joints(current_joint)
