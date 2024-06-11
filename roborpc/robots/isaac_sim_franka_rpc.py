import base64
from typing import Dict, List, Union

import cv2
import numpy as np
import requests
import zerorpc

from roborpc.cameras.camera_utils import base64_rgb, base64_depth
from robot_base import RobotBase
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config
from thirty_part.realman.robotic_arm import *


class IsaacSimFrankaRpc(RobotBase):

    def __init__(self, robot_id: str, ip_address: str, rpc_port: str):
        super().__init__()
        self.robot_id = robot_id
        self.ip_address = ip_address
        self.robot = None
        self.rpc_port = rpc_port
        self.last_arm_state = None
        self.robot_arm_dof = None
        self.robot_gripper_dof = None

    @staticmethod
    def http_call(command: str = "startup_isaac"):
        command_startup = {
            'command': 'startup_isaac'
        }
        command_shutdown = {
            'command': 'shutdown_isaac'
        }

        command_load_task_simulation = {
            'scene_task_id': '12312313',
            'command': 'load_task_simulation',
            'scene_usd': '/home/eai/Dev-Code/franka_isaac/scenes/lab_colmo_cab_can_1_7_auto_open.usd',
            'scene_task_name': 'pick_and_place_into_receptacle',
            'source_object_prim_path': '/root/Can_27',
            'target_object_prim_path': '/Mobile_Robot_Gen_2/Tray_Arm/Tray_Collider',
            'robot_init_pose': [352.0, -79.0, 0.0, 0.0, 0.0, -90.0],
            'task_command': 'pick Can_2 and place into Plate_1',
            'robot_init_arm_joints': [1.5710250036267028,
                                      -1.3962033283540387,
                                      -1.0474870365975733,
                                      2.7923556854543814,
                                      -0.5235441609572392,
                                      -3.302728092494427e-5,
                                      -1.5707987427375252
                                      ],
            'robot_init_gripper_degrees': [0.0, 0.0, -20.0, -75.0, 20.0, 75.0, 20.0, 75.0],
            'robot_init_body_position': [22.0, 0.0],
            'collector_path': '/home/jz08/Log'
        }
        command_retrieve_object_placement_info = {
            'command': 'retrieve_object_placement_info',
            'type': 'on',
            'source_obj': 'Can_2',
            'target_obj': 'Plate_1'
        }
        command_retrieve_object_bbox_info = {
            'command': 'retrieve_object_bbox_info',
            'object_name': 'Can_2'
        }

        command_check_source_on_target_aabb = {'command': 'check_source_on_target_aabb'}
        command_check_source_in_target_aabb = {'command': 'check_source_in_target_aabb'}
        command_check_pick_object = {'command': 'check_pick_object'}
        command_check_object_into_roi = {'command': 'check_object_into_roi'}
        command_check_object_near_object = {'command': 'check_object_near_object'}
        command_check_object_into_receptacle = {'command': 'check_object_into_receptacle'}

        if command == "startup_isaac":

            post_response1 = requests.post('http://127.0.0.1:6026/', json=command_startup)
            logger.info(post_response1.status_code)
            logger.info(post_response1.json())

            post_response2 = requests.post('http://127.0.0.1:6028/', json=command_load_task_simulation)
            logger.info(post_response2.json())

        elif command == "shutdown_isaac":
            post_response1 = requests.post('http://127.0.0.1:6026/', json=command_shutdown)
            logger.info(post_response1.status_code)
            logger.info(post_response1.json())

    def connect(self):
        self.http_call(command='startup_isaac')
        self.robot = zerorpc.Client(heartbeat=20)
        self.robot.connect("tcp://" + self.ip_address + ":" + self.rpc_port)

    def disconnect(self):
        self.http_call(command="shutdown_isaac")

    def get_robot_ids(self) -> List[str]:
        pass

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robot.set_ee_pose()

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        self.robot.set_joints()

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robot.set_gripper()

    def get_robot_state(self) -> Dict[str, List[float]]:
        return self.robot.get_robot_state()

    def get_dofs(self) -> int:
        return self.robot_arm_dof

    def get_joint_positions(self) -> List[float]:
        return self.robot.get_joint_positions()

    def get_gripper_position(self) -> List[float]:
        return self.robot.get_gripper_position()

    def get_joint_velocities(self) -> List[float]:
        return self.robot.get_joint_velocities()

    def get_ee_pose(self) -> List[float]:
        return self.robot.get_ee_pose()

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        """
        Get the intrinsics of the camera.
        """
        return self.robot.get_camera_intrinsics()

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        """
        Get the extrinsics of the camera.
        """
        return self.robot.get_camera_extrinsics()

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        """Read a frame from the camera.
        """
        sim_camera_info = self.robot.read_camera()
        camera_info = {"image": {
            "sim_handeye_rgb": base64_rgb(sim_camera_info["sim_handeye_rgb"]),
            "sim_varied_1_rgb": base64_rgb(sim_camera_info["sim_varied_1_rgb"]),
            "sim_varied_2_rgb": base64_rgb(sim_camera_info["sim_varied_2_rgb"]),
        }, "depth": {
            "sim_handeye_depth": base64_depth(sim_camera_info["sim_handeye_depth"]),
            "sim_varied_1_depth": base64_depth(sim_camera_info["sim_varied_1_depth"]),
            "sim_varied_2_depth": base64_depth(sim_camera_info["sim_varied_2_depth"]),
        }}
        return camera_info


class MultiIsaacSimFrankaRpc(RobotBase):
    def __init__(self):
        self.robot_config = config["roborpc"]["robots"]["isaac_sim_franka"]
        self.robots = None

    def connect(self):
        robot_ids = self.robot_config["robot_ids"]
        self.robots = {}
        for idx, robot_id in enumerate(robot_ids):
            ip_address = self.robot_config["ip_address"][idx]
            self.robots[robot_id] = SimFrankaRpc(robot_id, ip_address)
            self.robots[robot_id].connect()
            logger.success(f"RealMan Robot {robot_id} Connect Success!")

    def disconnect(self):
        for robot_id, robot in self.robots:
            robot.disconnect()
            logger.info(f"RealMan Robot {robot_id} Disconnect Success!")

    def get_robot_ids(self) -> List[str]:
        return self.robot_config["robot_ids"]

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        for robot_id, robot in self.robots.items():
            robot.set_ee_pose(action, action_space, blocking)

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        for robot_id, robot in self.robots.items():
            robot.set_joints(action, action_space, blocking)

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
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



