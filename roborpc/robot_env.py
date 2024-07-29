import os.path
import threading
import time
from pathlib import Path
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
        self.robots.connect_now()
        self.cameras.connect_now()

        self.env_update_rate = config['roborpc']['robot_env']['env_update_rate']
        self.robot_ids = self.robots.get_robot_ids()
        self.camera_ids = self.cameras.get_device_ids()
        for robot_id in self.robot_ids:
            if robot_id.startswith('realman'):
                self.kinematic_solver = CuroboSolverKinematic('realman')
            elif robot_id.startswith('panda'):
                self.kinematic_solver = CuroboSolverKinematic('panda')

        self.use_controller = config['roborpc']['robot_env']['use_controller']
        if self.use_controller:
            self.controllers = ComposedMultiController(kinematic_solver=self.kinematic_solver)
            self.controllers.connect_now()

    def __del__(self):
        self.robots.disconnect_now()
        self.cameras.disconnect_now()
        if self.use_controller:
            self.controllers.disconnect_now()

    def step(self, action: Dict[str, Dict[str, List[float]]],
             blocking: Union[bool, Dict[str, List[bool]]] = False) -> Dict[str, Dict[str, List[float]]]:
        blocking_info, action_info, new_action = {}, {}, {}
        for action_id, action_space_and_action in action.items():
            blocking_info = {action_id: {}}
            action_info = {action_id: {}}
            new_action = {action_id: {}}
            for action_space_id, action_value in action_space_and_action.items():
                if action_space_id == 'cartesian_position':
                    action_info[action_id].update({'cartesian_position': action_value})
                    pose = {action_id: action_value}
                    result = self.kinematic_solver.inverse_kinematics(pose)
                    if result is None:
                        print(f"Can't inverse kinematics for {action_id} with {action_value}")
                        result = self.robots.get_robot_state()[action_id]['joint_position']
                    else:
                        result = result[action_id]
                    action_info[action_id].update({'joint_position': result})
                    new_action[action_id].update({'joint_position': result})
                    if blocking is True:
                        blocking_info[action_id].update({'joint_position': True})
                    else:
                        blocking_info[action_id].update({'joint_position': False})
                elif action_space_id == 'joint_position':
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
                    new_action[action_id].update({'joint_position': new_action_value})
                if action_space_id == 'gripper_position':
                    if blocking is True:
                        blocking_info[action_id].update({'gripper_position': True})
                    action_info[action_id].update({'gripper_position': action_value})
                    new_action[action_id].update({'gripper_position': action_value})
        self.robots.set_robot_state(new_action, blocking_info)
        return action_info

    def reset(self, random_reset=False):
        self.robots.reset_robot_state()
        return self.get_observation()

    def get_observation(self):
        return self.robots.get_robot_state(), self.cameras.read_camera()

    def collect_data(self):
        pass
