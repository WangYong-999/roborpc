import argparse
import pickle
import threading
import traceback
from collections import OrderedDict
from typing import Union, Dict, List

import numpy as np

from roborpc.sim_robots.sim_robot_interface import SimRobotInterface, SimRobotRpcInterface
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config

import robosuite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper


class SimMujocoRobot(SimRobotInterface):

    def __init__(self):
        super().__init__()

        self.obs = {}
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
        self.blocking = {}
        self.robot_state = {}
        self.gripper_raw_open_close_range = {}
        self.gripper_normalized_open_close_range = {}
        self.robot_ids = self.robot_config['robot_ids'][0]
        self.robot_arm_dof = {}
        self.robot_gripper_dof = {}
        self.robot_zero_action = {}
        for robot_id in self.robot_ids:
            logger.info(f"Loading robot {robot_id}")
            self.robot_arm_dof[robot_id] = self.robot_config[robot_id]['robot_arm_dof']
            self.robot_gripper_dof[robot_id] = self.robot_config[robot_id]['robot_gripper_dof']
            self.robot_zero_action[robot_id] = self.robot_config[robot_id]['robot_zero_action']
            self.gripper_raw_open_close_range[robot_id] = self.robot_config[robot_id]['gripper_raw_open_close_range']
            self.gripper_normalized_open_close_range[robot_id] = self.robot_config[robot_id][
                'gripper_normalized_open_close_range']

        self.camera_ids = self.robot_config['camera_ids'][0]
        self.viewports = []
        self.cameras_xform = {}
        self.cameras_cache = {}
        for camera_id in self.camera_ids:
            self.cameras_cache[camera_id] = {}
            resolution = (self.robot_config[camera_id]['resolution'][0], self.robot_config[camera_id]['resolution'][1])
            self.viewports.append(self.robot_config[camera_id]['viewports'])
            self.cameras_cache[camera_id]['color'] = np.zeros((resolution[0], resolution[1], 3), dtype=np.float32)
            self.cameras_cache[camera_id]['depth'] = np.zeros((resolution[0], resolution[1]), dtype=np.float32)

        self.env = robosuite.make(
            env_name='NutAssemblySquare',
            robots='Panda',
            controller_configs=load_controller_config(default_controller='JOINT_POSITION'),
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera="frontview",
            ignore_done=True,
            camera_names=[*self.viewports],
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

        robot_rpc = SimRobotRpcInterface(self)

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
            for i, robot_id in enumerate(self.robot_ids):
                self.obs, _, _, _ = self.env.step(np.asarray(self.robot_config[robot_id]['robot_zero_action']),
                                                  set_qpos=self.robot_config[robot_id]['robot_zero_action'])
        task_completion_hold_count = -1
        n_actions = 1000
        actions = []

        while True:
            self.env.render()
            for i, robot_id in enumerate(self.robot_ids):
                robot = self.robot_state.get(robot_id, None)
                if robot is None:
                    continue
                robot_arm_dof = self.robot_arm_dof[robot_id]
                robot_gripper_dof = self.robot_gripper_dof[robot_id]
                arm_actions = robot.get('joint_position', None)
                normalized_gripper_actions = robot.get('gripper_position', None)
                # base_actions = self.robot_state.get('base_position', np.zeros(3))
                # torso_actions = self.robot_state.get('torso_position', np.zeros(1))
                # mode_action = self.robot_state.get('mode', [-1])
                # self.env.robots[0].enable_parts(right=True, left=True)
                if arm_actions is not None and normalized_gripper_actions is not None:
                    gripper_raw_open_close_range = self.gripper_raw_open_close_range[robot_id]
                    gripper_normalized_open_close_range = self.gripper_normalized_open_close_range[robot_id]
                    gripper_actions = (normalized_gripper_actions[0] - gripper_raw_open_close_range[0]) / (
                            gripper_raw_open_close_range[1] - gripper_raw_open_close_range[0]) * (
                                              gripper_normalized_open_close_range[1] -
                                              gripper_normalized_open_close_range[0]) + \
                                      gripper_normalized_open_close_range[0]
                    actions[:robot_arm_dof] = arm_actions
                    actions[robot_arm_dof:robot_arm_dof + 1] = np.asarray([gripper_actions])

                    self.obs, _, _, _ = self.env.step(actions, set_qpos=actions)
                    if self.env._check_success():
                        if task_completion_hold_count > 0:
                            task_completion_hold_count -= 1
                        else:
                            task_completion_hold_count = 10
                    else:
                        task_completion_hold_count = -1

    def connect_now(self):
        logger.info("Connect SimMujocoRobot")
        pass

    def disconnect_now(self):
        self.env.close()

    def get_robot_ids(self) -> List[str]:
        return self.robot_config["robot_ids"][0]

    def reset_robot_state(self):
        pass

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
        robot_state = {}
        for i, robot_id in enumerate(self.robot_ids):
            robot_state[robot_id] = {}
            robot_arm_dof = self.robot_arm_dof[robot_id]
            robot_gripper_dof = self.robot_gripper_dof[robot_id]
            joint_positions = self.obs["robot0_joint_pos"][:robot_arm_dof]
            gripper_positions = self.obs['robot0_gripper_qpos'][0]
            # gripper_raw_open_close_range -> gripper_normalized_open_close_range
            gripper_normalized_open_close_range = self.gripper_normalized_open_close_range[robot_id]
            gripper_raw_open_close_range = self.gripper_raw_open_close_range[robot_id]
            normalized_gripper_position = (gripper_positions - gripper_normalized_open_close_range[0]) / (
                    gripper_normalized_open_close_range[1] - gripper_normalized_open_close_range[0]) * (
                                                  gripper_raw_open_close_range[1] - gripper_raw_open_close_range[0]) + \
                                          gripper_raw_open_close_range[0]
            robot_state[robot_id] = {"joint_position": joint_positions.tolist(),
                                     "gripper_position": [normalized_gripper_position]}
        return robot_state

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

    def read_camera(self) -> Dict[str, Dict[str, bytes]]:
        camera_data = {}
        for camera_id, viewport in zip(self.camera_ids, self.viewports):
            camera_data[camera_id] = {}
            try:
                color_data = self.obs["{}_image".format(viewport)][::-1]
                camera_data[camera_id]['color'] = pickle.dumps(color_data)
                depth_data = self.obs["{}_depth".format(viewport)][::-1]
                camera_data[camera_id]['depth'] = pickle.dumps(depth_data)
                # self.cameras_cache[camera_id]['color'] = color_data
                # self.cameras_cache[camera_id]['depth'] = depth_data
            except Exception as e:
                logger.error(e)
                camera_data[camera_id]['color'] = pickle.dumps(self.cameras_cache[camera_id]['color'])
                camera_data[camera_id]['depth'] = pickle.dumps(self.cameras_cache[camera_id]['depth'])
        return camera_data


if __name__ == '__main__':
    SimMujocoRobot()
