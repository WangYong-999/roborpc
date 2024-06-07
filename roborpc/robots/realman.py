from typing import Dict, List, Union

from robot_base import RobotBase
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config
from thirty_part.realman.robotic_arm import *


class RealMan(RobotBase):

    def __init__(self, robot_id: str, ip_address: str):
        super().__init__()
        self.robot_id = robot_id
        self.ip_address = ip_address
        self.robot = None
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None

    def connect(self):
        def mcallback(data):
            logger.info("MCallback MCallback MCallback")
            # 判断接口类型
            if data.codeKey == MOVEJ_CANFD_CB:  # 角度透传
                print("透传结果:", data.errCode)
                print("当前角度:", data.joint[0], data.joint[1], data.joint[2], data.joint[3], data.joint[4],
                      data.joint[5])
            elif data.codeKey == MOVEP_CANFD_CB:  # 位姿透传
                print("透传结果:", data.errCode)
                print("当前角度:", data.joint[0], data.joint[1], data.joint[2], data.joint[3], data.joint[4],
                      data.joint[5])
                print("当前位姿:", data.pose.position.x, data.pose.position.y, data.pose.position.z, data.pose.euler.rx,
                      data.pose.euler.ry, data.pose.euler.rz)
            elif data.codeKey == FORCE_POSITION_MOVE_CB:  # 力位混合透传
                print("透传结果:", data.errCode)
                print("当前力度：", data.nforce)

        callback = CANFD_Callback(mcallback)
        robot_type = RM75
        self.robot = Arm(robot_type, self.ip_address, callback)
        if robot_type == RM75:
            self.robot_arm_dof = 7
        elif robot_type == RM65:
            self.robot_arm_dof = 6
        else:
            assert False, "Unsupported robot type"
        self.robot_gripper_dof = 1
        self.last_arm_state = [0.0] * self.robot_arm_dof

        # API版本信息
        logger.info(self.robot.API_Version())

    def disconnect(self):
        self.robot.RM_API_UnInit()
        self.robot.Arm_Socket_Close()

    def get_robot_ids(self) -> List[str]:
        pass

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "cartesian_position", blocking: Union[bool, List[bool]] = False):
        pass

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        pass

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]], action_space: Union[str, List[str]] = "gripper_position", blocking: Union[bool, List[bool]] = False):
        pass

    def get_robot_state(self) -> Dict[str, List[float]]:
        pass

    def get_dofs(self) -> int:
        return self.robot_arm_dof

    def get_joint_positions(self) -> List[float]:
        arm_state = self.robot.Get_Current_Arm_State(retry=1)
        if arm_state[0] == 0:
            self.last_arm_state = arm_state
            return arm_state[1]
        else:
            logger.error("RealMan Robot Get Joint Positions Failed!")
            return self.last_arm_state[1]

    def get_gripper_position(self) -> List[float]:
        ret, gripper_state = self.robot.Get_Gripper_State()
        return gripper_state

    def get_joint_velocities(self) -> List[float]:
        pass

    def get_ee_pose(self) -> List[float]:
        arm_state = self.robot.Get_Current_Arm_State(retry=1)
        if arm_state[0] == 0:
            self.last_arm_state = arm_state
            return arm_state[2]
        else:
            return self.last_arm_state[2]


class MultiRealMan(RobotBase):
    def __init__(self):
        self.robot_config = config["roborpc"]["robots"]["realman"]
        self.robots = None

    def connect(self):
        robot_ids = self.robot_config["robot_ids"]
        self.robots = {}
        for idx, robot_id in enumerate(robot_ids):
            ip_address = self.robot_config["ip_address"][idx]
            self.robots[robot_id] = RealMan(robot_id, ip_address)
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

