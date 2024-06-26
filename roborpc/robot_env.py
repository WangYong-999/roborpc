import threading
import time
from typing import Dict, List, Union

import gym
from roborpc.common.config_loader import config

from roborpc.robots.composed_multi_robots import ComposedMultiRobots
from roborpc.cameras.composed_multi_cameras import ComposedMultiCameras
from roborpc.controllers.composed_multi_controllers import ComposedMultiController
from roborpc.kinematics_solver.curobo_solver_kinematic import CuroboSolverKinematic


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

        self.kinematic_solver = CuroboSolverKinematic()
        self.env_update_rate = config['roborpc']['robot_env']['env_update_rate']
        self.robot_ids = self.robots.get_robot_ids()
        self.camera_ids = self.cameras.get_device_ids()

    def __del__(self):
        self.robots.disconnect_now()
        self.cameras.disconnect_now()
        self.controllers.disconnect_now()

    def step(self, action: Dict[str, Dict[str, List[float]]],
             blocking: Union[bool, Dict[str, List[bool]]] = False) -> Dict[str, Dict[str, List[float]]]:
        action_info = {}
        blocking_info = {}
        for action_id, action_space_and_action in action.items():
            for action_space_id, action_value in action_space_and_action.items():
                if action_space_id == 'cartesian_position':
                    if blocking is True:
                        blocking_info[action_id] = {'cartesian_position': True}
                    action_info[action_id] = {'cartesian_position': action_value}
                    pose = {action_id: action_value}
                    action_info[action_id] = {'joint_position': self.kinematic_solver.inverse_kinematics(pose)}
                if action_space_id == 'joint_position':
                    if blocking is True:
                        blocking_info[action_id] = {'joint_position': True}
                    action_info[action_id] = {'joint_position': action_value}
                    joints_angle = {action_id: action_value}
                    action_info[action_id] = {
                        'cartesian_position': self.kinematic_solver.forward_kinematics(joints_angle)}
                if action_space_id == 'gripper_position':
                    if blocking is True:
                        blocking_info[action_id] = {'gripper_position': True}
                    action_info[action_id] = {'gripper_position': action_value}
        self.robots.set_robot_state(action, blocking_info)
        return action_info

    def reset(self, random_reset: bool = False):
        pass

    def get_observation(self):
        observation = {}
        observation.update(self.robots.get_robot_state())
        observation.update(self.cameras.read_camera())
        return observation

    def collect_data(self):
        pass
