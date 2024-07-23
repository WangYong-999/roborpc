import os
import pickle
import random
import time
from pathlib import Path

from omni.isaac.kit import SimulationApp

from roborpc.common.config_loader import config
from roborpc.kinematics_solver.trajectory_interpolation import action_linear_interpolation

simulation_app = SimulationApp({"headless": False, "open_usd":
    os.path.join(Path(__file__).parent, f"usds/robots/{config['roborpc']['sim_robots']['isaac_sim']['robot_usd_path']}")})

import traceback
import omni
from omni.isaac.sensor import Camera
from pxr import Gf, UsdGeom
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core import World
from tasks.pick_place import PickPlace
from controllers.pick_place import PickPlaceController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.debug_draw import _debug_draw

import threading
import transforms3d as t3d
import base64
import numpy as np
import cv2
from pynput import keyboard
from pynput.keyboard import Key
from typing import List, Union, Dict
from roborpc.sim_robots.sim_robot_interface import SimRobotInterface, SimRobotRpcInterface
from roborpc.common.logger_loader import logger


class SimIsaacRobot(SimRobotInterface):
    def __init__(self):
        self.blocking = {}
        self.robot_state = {}
        self.gripper_raw_open_close_range = {}
        self.gripper_normalized_open_close_range = {}
        self.my_world = World(stage_units_in_meters=1.0)
        self.button_pressed = False
        self.reset_robot_flag = False
        self.reset_world_flag = False

        self.env_update_rate = config['roborpc']['robot_env']['env_update_rate']
        self.robot_config = config['roborpc']['sim_robots']['isaac_sim']
        self.robot_ids = self.robot_config['robot_ids'][0]
        self.robot_arm_dof = {}
        self.robot_gripper_dof = {}
        self.robot_zero_action = {}
        self.my_controller = {}
        self.articulation_controller = {}
        self.action_dof = {}
        self.gripper_signs = {}
        for robot_id in self.robot_ids:
            logger.info(f"Loading robot {robot_id}")
            self.gripper_signs[robot_id] = self.robot_config[robot_id]['gripper_signs']
            self.robot_arm_dof[robot_id] = self.robot_config[robot_id]['robot_arm_dof']
            self.robot_gripper_dof[robot_id] = self.robot_config[robot_id]['robot_gripper_dof']
            self.robot_zero_action[robot_id] = self.robot_config[robot_id]['robot_zero_action']
            self.action_dof[robot_id] = self.robot_config[robot_id]['action_dof']
            cube_initial_position_x = np.random.uniform(0.4, 0.45)
            cube_initial_position_y = np.random.uniform(0.4, 0.5)
            my_task = PickPlace(name=robot_id,
                                cube_initial_position=np.array([cube_initial_position_x, cube_initial_position_y, 0.5]),
                                gripper_base_link_prim_path=self.robot_config[robot_id]['gripper_base_link_prim_path'],
                                gripper_joint_prim_names=self.robot_config[robot_id]['gripper_joint_prim_names'],
                                arm_base_link_prim_path=self.robot_config[robot_id]['arm_base_link_prim_path'],
                                )
            self.my_world.add_task(my_task)
        self.my_world.reset()
        self.robots = {}
        self.stiffnesses = {}
        self.dampings = {}
        for robot_id in self.robot_ids:
            robot = self.my_world.scene.get_object(robot_id)
            self.robots[robot_id] = robot
            self.my_controller[robot_id] = PickPlaceController(name=robot_id, robot_articulation=robot,
                                                               gripper=robot.gripper,
                                                               robot_description_path=self.robot_config[robot_id][
                                                                   'robot_description_path'],
                                                               rmpflow_config_path=self.robot_config[robot_id][
                                                                   'rmpflow_config_path'],
                                                               urdf_path=self.robot_config[robot_id]['urdf_path'],
                                                               end_effector_frame_name=self.robot_config[robot_id][
                                                                   'end_effector_frame_name'],
                                                               )
            self.articulation_controller[robot_id] = robot.get_articulation_controller()

            self.gripper_raw_open_close_range[robot_id] = self.robot_config[robot_id]['gripper_raw_open_close_range']
            self.gripper_normalized_open_close_range[robot_id] = self.robot_config[robot_id][
                'gripper_normalized_open_close_range']
            self.stiffnesses[robot_id] = self.robot_config[robot_id]['stiffnesses']
            self.dampings[robot_id] = self.robot_config[robot_id]['dampings']

        self.camera_ids = self.robot_config['camera_ids'][0]
        self.viewports = {}
        self.cameras_xform = {}
        self.cameras_cache = {}
        for camera_id in self.camera_ids:
            self.cameras_cache[camera_id] = {}
            camera_prim_path = self.robot_config[camera_id]['camera_prim_path']
            frequency = self.robot_config[camera_id]['frequency']
            resolution = (self.robot_config[camera_id]['resolution'][0], self.robot_config[camera_id]['resolution'][1])
            self.cameras_xform[camera_id] = XFormPrim(prim_path=camera_prim_path)
            self.viewports[camera_id] = Camera(
                prim_path=camera_prim_path,
                frequency=frequency,
                resolution=resolution,
                # position=np.array([-0.07295, 0.03137, 0.00037]),
                # orientation=rot_utils.euler_angles_to_quats(np.array([122.225, -18.567, -167.487]), degrees=True),
            )
            self.cameras_cache[camera_id]['color'] = np.zeros((resolution[0], resolution[1], 3), dtype=np.float32)
            self.cameras_cache[camera_id]['depth'] = np.zeros((resolution[0], resolution[1]), dtype=np.float32)

        self.draw = _debug_draw.acquire_debug_draw_interface()
        robot_rpc = SimRobotRpcInterface(self)

        def _listener_rpc() -> None:
            try:
                import zerorpc
                s = zerorpc.Server(robot_rpc)
                rpc_port = self.robot_config['sever_rpc_ports'][0]
                s.bind(f"tcp://0.0.0.0:{rpc_port}")
                logger.info(f"Starting RPC server on port {rpc_port}")
                s.run()
            except (Exception,):
                logger.error('Error in DaemonLauncher._listener_rpc: %s' % traceback.format_exc())

        threading.Thread(target=self.run_key_listen).start()
        threading.Thread(target=self.random_scene_objects_pose, name='RandomObjectsPose', daemon=False).start()
        threading.Thread(target=_listener_rpc, name='RpcListener', daemon=False).start()

        self._sim_loop()

    def _sim_loop(self):
        for viewport_id, viewport in self.viewports.items():
            viewport.initialize()
            viewport.add_distance_to_image_plane_to_frame()
        while simulation_app.is_running():
            try:
                if self.button_pressed:
                    self.my_world.reset()
                    self.reset_world_flag = True
                self.my_world.step(render=True)
                if self.my_world.is_playing():
                    start_time = time.time()
                    self.obs = self.my_world.get_observations()
                    if self.my_world.current_time_step_index <= 3 or self.reset_robot_flag:
                        logger.info("Resetting robots")
                        for i, robot_id in enumerate(self.robot_ids):
                            self.my_controller[robot_id].reset()
                            self.robots[robot_id]._articulation_view.initialize()
                            # self.robots[robot_id]._articulation_view.set_gains(kps=self.stiffnesses[robot_id], kds=self.dampings[robot_id])
                        self._reset_robot()
                        self.reset_robot_flag = False
                    if self.my_world.current_time_step_index < 5:
                        continue
                    for camera_id in self.camera_ids:
                        try:
                            self.cameras_cache[camera_id]['color'] = np.asarray(self.viewports[camera_id].get_rgba(), dtype=np.uint8)
                            self.cameras_cache[camera_id]['depth'] = np.asarray(self.viewports[camera_id].get_depth(), dtype=np.uint8)
                        except Exception as e:
                            logger.info(f"Error getting camera {camera_id} data: {e}")
                    for i, robot_id in enumerate(self.robot_ids):
                        start_point = self.obs[robot_id]["end_effector_position"]
                        # TODO: Check if end effector position is too close to ground
                        if start_point[2] < 0.12:
                            logger.info(f"End effector position is too close to ground for {robot_id}")
                            continue
                        # end_point = start_point + np.array([0, 0, -1])
                        # point_list_1 = [(start_point[0], start_point[1], start_point[2])]
                        # point_list_2 = [(end_point[0], end_point[1], end_point[2])]
                        # colors = [(0, 0, 1, 1)]
                        # sizes = [1]
                        # self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)
                        robot = self.robot_state.get(robot_id, None)
                        if robot is None:
                            continue
                        robot_arm_dof = self.robot_arm_dof[robot_id]
                        robot_gripper_dof = self.robot_gripper_dof[robot_id]
                        articulation_actions = [None] * (robot_arm_dof + robot_gripper_dof)
                        actions = [None] * (robot_arm_dof + robot_gripper_dof)
                        arm_actions = robot.get("joint_position", np.nan * np.ones(robot_arm_dof))
                        normalized_gripper_actions = robot.get("gripper_position", np.nan * np.ones(robot_gripper_dof))
                        gripper_raw_open_close_range = self.gripper_raw_open_close_range[robot_id]
                        gripper_normalized_open_close_range = self.gripper_normalized_open_close_range[robot_id]
                        gripper_actions = (normalized_gripper_actions[0] - gripper_raw_open_close_range[0]) / (
                                gripper_raw_open_close_range[1] - gripper_raw_open_close_range[0]) * (
                                                  gripper_normalized_open_close_range[1] -
                                                  gripper_normalized_open_close_range[0]) + \
                                          gripper_normalized_open_close_range[0]
                        actions[:robot_arm_dof] = arm_actions
                        actions[robot_arm_dof:robot_arm_dof + robot_gripper_dof] = np.asarray(
                            [gripper_actions * self.gripper_signs[robot_id][0],
                             gripper_actions * self.gripper_signs[robot_id][1]])
                        self.articulation_controller[robot_id].apply_action(ArticulationAction(joint_positions=actions))
                        # if gripper_actions < 0.04:
                        #     self.articulation_controller[robot_id].apply_action(ArticulationAction(joint_positions=actions))
                        # else:
                        #     self.robots[robot_id].set_joint_positions(actions)
                        self.robot_state = {}
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        assert 1 / elapsed_time >= self.env_update_rate, f"Isaac Sim Update rate is too slow. Turn down environment update rate ."
                        logger.info(f"Elapsed time: {elapsed_time / 1000} ms, Update rate: {1 / elapsed_time} Hz")
            except (Exception,):
                logger.error('Error in SimIsaacRobot._sim_loop: %s' % traceback.format_exc())
                break
        simulation_app.close()

    def _reset_robot(self):
        state = {}
        robot_obs = self.get_robot_state()

        for i, robot_id in enumerate(self.robot_ids):
            robot_arm_dof = self.robot_arm_dof[robot_id]
            robot_gripper_dof = self.robot_gripper_dof[robot_id]

            arm_actions = self.robot_zero_action[robot_id][:robot_arm_dof]
            gripper_actions = self.robot_zero_action[robot_id][robot_arm_dof:robot_arm_dof + robot_gripper_dof]
            state[robot_id] = {'joint_position': arm_actions, 'gripper_position': gripper_actions}

        new_state = action_linear_interpolation(robot_obs, state)
        for robot_id, new_robot_state in new_state.items():
            robot_arm_dof = self.robot_arm_dof[robot_id]
            robot_gripper_dof = self.robot_gripper_dof[robot_id]

            arm_actions = new_robot_state['joint_position']
            gripper_actions = new_robot_state['gripper_position']
            gripper_raw_open_close_range = self.gripper_raw_open_close_range[robot_id]
            gripper_normalized_open_close_range = self.gripper_normalized_open_close_range[robot_id]
            gripper_actions = (gripper_actions[0] - gripper_raw_open_close_range[0]) / (
                    gripper_raw_open_close_range[1] - gripper_raw_open_close_range[0]) * (
                                      gripper_normalized_open_close_range[1] -
                                      gripper_normalized_open_close_range[0]) + \
                              gripper_normalized_open_close_range[0]
            for arm_action in arm_actions:
                actions = [None] * (robot_arm_dof + robot_gripper_dof)
                actions[:robot_arm_dof] = arm_action
                actions[robot_arm_dof:robot_arm_dof + robot_gripper_dof] = np.asarray(
                    [gripper_actions * self.gripper_signs[robot_id][0],
                     gripper_actions * self.gripper_signs[robot_id][1]])
                self.robots[robot_id].set_joint_positions(actions)

        logger.info(f"Reset robot state: {state}")

    def run_key_listen(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        try:
            if key == Key.down:
                self.button_pressed = True
        except AttributeError:
            logger.error(f"Unknown key {key}")

    def on_release(self, key):
        try:
            if key == Key.down:
                self.button_pressed = False
        except AttributeError:
            logger.error(f"Unknown key {key}")

    def random_scene_objects_pose(self):
        """
        Set the pose of the objects in the scene.
        """
        while True:
            if self.reset_world_flag:
                logger.info("Resetting scene objects")
                cube_initial_position_x = np.random.uniform(0.3, 0.6)
                cube_initial_position_y = np.random.uniform(-0.4, 0.4)
                cube_initial_rotation_z = np.random.uniform(-180, 180)
                cube_initial_position = np.array([cube_initial_position_x, cube_initial_position_y, 0.1, 0, 0, cube_initial_rotation_z])
                objs_pose = {'/World/_06_mustard_bottle': cube_initial_position, '/World/Cube': np.array([5, 5, 0.1, 0, 0, 0])}
                stage = omni.usd.get_context().get_stage()
                for obj_path, pose in objs_pose.items():
                    object_prim = stage.GetPrimAtPath(obj_path)
                    if not object_prim.GetAttribute("xformOp:translate"):
                        UsdGeom.Xformable(object_prim).AddTranslateOp()
                    if not object_prim.GetAttribute("xformOp:rotateXYZ"):
                        UsdGeom.Xformable(object_prim).AddRotateXYZOp()
                    object_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(pose[0], pose[1], pose[2]))
                    object_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(pose[3], pose[4], pose[5]))
                omni.kit.app.get_app().next_update_async()
                self.reset_world_flag = False
            else:
                time.sleep(0.1)

    def connect_now(self):
        logger.info("Connected to robot")
        pass

    def disconnect_now(self):
        pass

    def get_robot_ids(self) -> List[str]:
        return self.robot_config["robot_ids"][0]

    def reset_robot_state(self):
        self.reset_robot_flag = True

    def set_robot_state(self, state: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
                        blocking: Union[Dict[str, bool], Dict[str, Dict[str, bool]]]):
        self.robot_state = state
        self.blocking = blocking

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
        robot_state = {}
        for i, robot_id in enumerate(self.robot_ids):
            robot_state[robot_id] = {}
            robot_arm_dof = self.robot_arm_dof[robot_id]
            robot_gripper_dof = self.robot_gripper_dof[robot_id]
            joint_positions = self.obs[robot_id]["joint_positions"][:robot_arm_dof]
            gripper_positions = self.obs[robot_id]["joint_positions"][robot_arm_dof:robot_arm_dof + 1]
            # gripper_raw_open_close_range -> gripper_normalized_open_close_range
            gripper_normalized_open_close_range = self.gripper_normalized_open_close_range[robot_id]
            gripper_raw_open_close_range = self.gripper_raw_open_close_range[robot_id]
            normalized_gripper_position = (gripper_positions[0] - gripper_normalized_open_close_range[0]) / (
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
        intrinsics = {}
        for camera_id, viewport in self.viewports.items():
            intrinsics[camera_id] = viewport.get_intrinsics_matrix().tolist()
        return intrinsics

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        cameras_extrinsics = {}
        for camera_id, camera_xform in self.cameras_xform.items():
            camera_world_pose = camera_xform.get_world_pose()
            camera_world_position = (camera_world_pose[0] / 100).tolist()  # Convert from cm to meter.
            camera_world_quaternions = (camera_world_pose[1]).tolist()  # Quaternion is scalar-first (w, x, y, z).
            camera_world_euler = t3d.euler.quat2euler(camera_world_quaternions, axes='sxyz')
            camera_world_pose_data = [*camera_world_position, *camera_world_euler]
            cameras_extrinsics[camera_id] = camera_world_pose_data
        return cameras_extrinsics

    def read_camera(self) -> Dict[str, Dict[str, bytes]]:
        camera_data = {}
        for camera_id, camera_cache in self.cameras_cache.items():
            color_data = cv2.cvtColor(camera_cache['color'], cv2.COLOR_RGBA2RGB)
            color_data_base = pickle.dumps(color_data)
            depth_data_base = pickle.dumps(camera_cache['depth'])
            camera_data[camera_id] = {"color": color_data_base, "depth": depth_data_base}
        return camera_data


SimIsaacRobot()
