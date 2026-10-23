from typing import Union, List, Dict

import numpy as np
import time

from roborpc.cameras.transformations import ur_axis_angle_to_quat
from roborpc.robots.robot_base import RobotBase
from roborpc.common.logger_loader import logger


class UR(RobotBase):

    def __init__(self, robot_id: str, ip_address: str):
        super().__init__()
        self.robot_id = robot_id
        self.ip_address = ip_address
        self.robot = None
        self.r_inter = None
        self.gripper = None
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None
        self._free_drive = False

    def connect_now(self):
        import rtde_control
        import rtde_receive
        from roborpc.robots.robotiq_gripper import RobotiqGripper
        try:
            self.robot = rtde_control.RTDEControlInterface(self.ip_address)
        except Exception as e:
            logger.error(f"Failed to connect to UR robot {self.robot_id} at {self.ip_address}: {e}")
        self.r_inter = rtde_receive.RTDEReceiveInterface(self.ip_address)
        self.gripper = RobotiqGripper()
        self.gripper.connect(hostname=self.ip_address, port=63352)
        self._free_drive = False
        self.robot.endFreedriveMode()
        logger.success("Connected to UR robot")
        pass

    def disconnect_now(self):
        pass

    def get_robot_ids(self) -> List[str]:
        pass

    def reset_robot_state(self):
        pass

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        for action_space, action in state.items():
            if action_space == "joint_position":
                velocity = 0.5
                acceleration = 0.5
                dt = 1.0 / 500  # 2ms
                lookahead_time = 0.2
                gain = 100

                robot_joints = action
                t_start = self.robot.initPeriod()
                self.robot.servoJ(
                    robot_joints, velocity, acceleration, dt, lookahead_time, gain
                )
                self.robot.waitPeriod(t_start)
            elif action_space == "gripper_position":
                gripper_pos = action * 255
                self.gripper.move(int(gripper_pos), 255, 10)
            elif action_space == "cartesian_position":
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
        robot_state = {"joint_position": self.get_joint_positions(),
                       "gripper_position": self.get_gripper_position(),
                       "cartesian_position": self.get_ee_pose()}
        return robot_state

    def get_dofs(self) -> Union[int, Dict[str, int]]:
        pass

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.r_inter.getActualQ()

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        xyz_rpy = self.r_inter.getActualTCPPose()
        pos_quat = xyz_rpy[:3] + ur_axis_angle_to_quat(xyz_rpy[3:])
        return pos_quat


