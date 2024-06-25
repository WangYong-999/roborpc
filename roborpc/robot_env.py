import threading
import time
from typing import Dict, List, Union

import gym
from roborpc.common.config_loader import config

from roborpc.robots.composed_multi_robots import ComposedMultiRobots
from roborpc.cameras.composed_multi_cameras import ComposedMultiCameras
from roborpc.controllers.composed_multi_controllers import ComposedMultiController


class RobotEnv(gym.Env):
    _single_lock = threading.Lock()
    _instance = None

    def __new__(cls):
        # single instance
        with RobotEnv._single_lock:
            RobotEnv._instance = object.__new__(cls)
            RobotEnv._instance._initialize()
            return RobotEnv._instance

    def _initialize(self):
        self.robots = ComposedMultiRobots()
        self.cameras = ComposedMultiCameras()
        self.controllers = ComposedMultiController()
        self.robots.connect_now()
        self.cameras.connect_now()
        self.controllers.connect_now()

        self.env_update_rate = config['roborpc']['robot_env']['env_update_rate']
        self.robot_ids = self.robots.get_robot_ids()
        self.camera_ids = self.cameras.get_device_ids()

    def __del__(self):
        self.robots.disconnect_now()
        self.cameras.disconnect_now()
        self.controllers.disconnect_now()

    def step(self, action: Union[List[float], Dict[str, List[float]]],
             action_space: Union[str, Dict[str, str]] = "cartesian_position_gripper_position",
             blocking: Union[bool, Dict[str, bool]] = False):
        arm_action_space = {}
        gripper_action_space = {}
        for action_id, action_name in action_space.items():
            if action_name == "cartesian_position_gripper_position":
                arm_action_space[action_id] = "cartesian_position"
                gripper_action_space[action_id] = "gripper_position"
            elif action_name == "joint_position_gripper_position":
                arm_action_space[action_id] = "joint_position"
                gripper_action_space[action_id] = "gripper_position"
            else:
                raise ValueError("Invalid action space")
        self.robots.set_ee_pose(action, action_space, blocking)
        self.robots.set_gripper(action, gripper_action_space, blocking)

    def reset(self, random_reset: bool = False):
        pass

    def get_observation(self):
        observation = {}
        observation.update(self.robots.get_robot_state())
        observation.update(self.cameras.read_camera())
        return observation

    def collect_data(self):
        pass
