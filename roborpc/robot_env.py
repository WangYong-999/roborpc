import threading
import time
from typing import Dict, List, Union

import gym
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger

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
        blocking_info, action_info = {}, {}
        for action_id, action_space_and_action in action.items():
            blocking_info = {action_id: {}}
            action_info = {action_id: {}}
            for action_space_id, action_value in action_space_and_action.items():
                if action_space_id == 'cartesian_position':
                    if blocking is True:
                        blocking_info[action_id].update({'cartesian_position': True})
                    action_info[action_id].update({'cartesian_position': action_value})
                    pose = {action_id: action_value}
                    action_info[action_id].update({'joint_position': self.kinematic_solver.inverse_kinematics(pose)})
                if action_space_id == 'joint_position':
                    if blocking is True:
                        blocking_info[action_id].update({'joint_position': True})
                    else:
                        blocking_info[action_id].update({'joint_position': False})
                    if isinstance(action_value[0], list):
                        new_action_value = action_value[-1]
                    else:
                        new_action_value = action_value
                    action_info[action_id].update({'joint_position': new_action_value})
                    joints_angle = {action_id: new_action_value}
                    action_info[action_id].update({'cartesian_position': self.kinematic_solver.forward_kinematics(joints_angle)[action_id]})
                if action_space_id == 'gripper_position':
                    if blocking is True:
                        blocking_info[action_id].update({'gripper_position': True})
                    action_info[action_id].update({'gripper_position': action_value})
        self.robots.set_robot_state(action, blocking_info)
        return action_info

    def reset(self, random_reset: bool = False):
        pass

    def get_observation(self):
        return self.robots.get_robot_state(), self.cameras.read_camera()

    def collect_data(self):
        pass
