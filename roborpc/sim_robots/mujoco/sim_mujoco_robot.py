import argparse
import threading
import traceback
from collections import OrderedDict
from typing import Union, Dict, List

import numpy as np

from roborpc.sim_robots.sim_robot_interface import SimRobotInterface
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config

import robosuite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper


class SimMujocoRobotRpc(SimRobotInterface):
    def __init__(self, robot):
        self.robot = robot

    def connect_now(self):
        pass

    def disconnect_now(self):
        pass

    def get_robot_ids(self) -> List[str]:
        return self.robot.get_robot_ids()

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        self.robot.set_robot_state(state, blocking)

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robot.set_ee_pose(action, action_space, blocking)

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        self.robot.set_joints(action, action_space, blocking)

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        self.robot.set_gripper(action, action_space, blocking)

    def get_robot_state(self) -> Dict[str, List[float]]:
        return self.robot.get_robot_state()

    def get_dofs(self) -> Dict[str, int]:
        return self.robot.get_dofs()

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_joint_positions()

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_gripper_position()

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_joint_velocities()

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robot.get_ee_pose()

    def get_device_ids(self) -> List[str]:
        return self.robot.get_device_ids()

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        return self.robot.get_camera_intrinsics()

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        return self.robot.get_camera_extrinsics()

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        return self.robot.read_camera()


class SimMujocoRobot(SimRobotInterface):

    def __init__(self):
        super().__init__()

        self.blocking = None
        self.robot_state = {}
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, help="task (choose among 100+ tasks)")
        parser.add_argument("--layout", type=int, help="kitchen layout (choose number 0-9)")
        parser.add_argument("--style", type=int, help="kitchen style (choose number 0-11)")
        args = parser.parse_args()

        # kitchen tasks now only support PandaMobile
        tasks = OrderedDict([
            ("PnPCounterToCab", "pick and place from counter to cabinet"),
            ("PnPCounterToSink", "pick and place from counter to sink"),
            ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
            ("PnPStoveToCounter", "pick and place from stove to counter"),
            ("OpenSingleDoor", "open cabinet or microwave door"),
            ("CloseDrawer", "close drawer"),
            ("TurnOnMicrowave", "turn on microwave"),
            ("TurnOnSinkFaucet", "turn on sink faucet"),
            ("TurnOnStove", "turn on stove"),
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
        ])

        self.robot_config = config['roborpc']['sim_robots']['mujoco']

        self.env = robosuite.make(
            env_name='NutAssemblySquare',
            robots='Panda',
            controller_configs=load_controller_config(default_controller='JOINT_POSITION'),
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=True,
            camera_depths=True,
            reward_shaping=True,
            control_freq=20,
            hard_reset=False,
        )
        self.env = VisualizationWrapper(self.env)
        self.env.viewer.set_camera(camera_id=0)
        self.env.reset()
        logger.success(f"Mujoco Task: {args.task}, Instruction: {self.env.get_ep_meta().get('lang', None)}")

        robot_rpc = SimMujocoRobotRpc(self)

        def _listener_rpc() -> None:
            try:
                import zerorpc
                s = zerorpc.Server(robot_rpc)
                rpc_port = self.robot_config['sever_rpc_ports'][0]
                s.bind(f"tcp://0.0.0.0:{rpc_port}")
                s.run()
            except (Exception,):
                logger.error('Error in DaemonLauncher._listener_rpc: %s' % traceback.format_exc())

        threading.Thread(target=_listener_rpc, name='RpcListener', daemon=False).start()
        logger.success(f"Mujoco Robot RPC Server started on port {self.robot_config['sever_rpc_ports'][0]}")

        self._sim_loop()

    def _sim_loop(self):
        for _ in range(100):
            self.env.render()
            self.obs, _, _, _ = self.env.step(np.asarray(self.robot_config['zero_action'][0]),
                                              set_qpos=self.robot_config['zero_action'][0])
        task_completion_hold_count = -1
        n_actions = 1000
        actions = 0.1 * np.random.uniform(low=-1.0, high=1.0, size=(n_actions, self.env.action_spec[0].shape[0]))

        while True:
            self.env.render()
            robot = self.robot_state.get('panda_1', None)
            if robot is None:
                continue
            arm_actions = robot.get('joint_position', None)
            gripper_actions = robot.get('gripper_position', None)
            # base_actions = self.robot_state.get('base_position', np.zeros(3))
            # torso_actions = self.robot_state.get('torso_position', np.zeros(1))
            # mode_action = self.robot_state.get('mode', [-1])
            # self.env.robots[0].enable_parts(right=True, left=True)
            if arm_actions is not None and gripper_actions is not None:
                print("Arm actions: ", arm_actions)
                print("Gripper actions: ", gripper_actions)
                # [0, 1] -> [-1, 1]
                grip = 2 * (gripper_actions[0] - 0.5)
                normalized_gripper_position = np.asarray([grip])
                action = np.concatenate([arm_actions, normalized_gripper_position])
                self.obs, _, _, _ = self.env.step(action, set_qpos=action)
                if self.env._check_success():
                    if task_completion_hold_count > 0:
                        task_completion_hold_count -= 1
                    else:
                        task_completion_hold_count = 10
                else:
                    task_completion_hold_count = -1

    def connect_now(self):
        pass

    def disconnect_now(self):
        self.env.close()

    def get_robot_ids(self) -> List[str]:
        return self.robot_config["robot_ids"][0]

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        self.robot_state = state
        self.blocking = blocking

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "cartesian_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        return {}

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, Dict[str, str]] = "joint_position",
                   blocking: Union[bool, Dict[str, bool]] = False):
        return {}

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "gripper_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        return {}

    def get_robot_state(self) -> Dict[str, List[float]]:
        # [-1, 1] -> [0, 1]
        normalized_gripper_position = (self.obs['robot0_gripper_qpos'][0] + 1) / 2
        obs = {
            'panda_1':
                {
                    "joint_position": self.obs['robot0_joint_pos'].tolist(),
                    "gripper_position": [normalized_gripper_position],
                    "ee_pose": np.concatenate([self.obs['robot0_eef_pos'], self.obs['robot0_eef_quat']]).tolist(),
                }
        }
        return obs

    def get_dofs(self) -> Union[int, Dict[str, int]]:
        return {}

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return {}

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        return {}

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        return {}

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        return {}

    def get_device_ids(self) -> List[str]:
        return self.robot_config["camera_ids"]

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        return {}

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        return {}

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        camera_name = 'agentview'
        image = self.obs["{}_image".format(camera_name)][::-1]
        depth_map = self.obs["{}_depth".format(camera_name)][::-1]
        return {camera_name: {"image": image, "depth_map": depth_map}}


if __name__ == '__main__':
    SimMujocoRobot()