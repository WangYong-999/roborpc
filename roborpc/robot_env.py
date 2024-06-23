import threading
import time
from typing import Dict, List

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

    def __del__(self):
        self.robots.disconnect_now()
        self.cameras.disconnect_now()
        self.controllers.disconnect_now()

    def step(self, action):
        pass

    def reset(self, random_reset: bool = False):
        pass

    def get_observation(self):
        observation = {}
        observation['robots'] = self.robots.get_robot_state()
        observation['cameras'] = self.cameras.read_camera()
        return observation

    def collect_data(self):
        pass

