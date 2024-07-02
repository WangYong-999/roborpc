from omni.isaac.kit import SimulationApp
CONFIG = {"width": 1280, "height": 720, "sync_loads": False,
          "headless": False, "renderer": "RayTracedLighting"}
simulation_app = SimulationApp(CONFIG)
import omni
from pxr import Gf, UsdGeom
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import is_stage_loading
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.numpy.rotations as rot_utils

import cv2
import traceback
import threading
import numpy as np
import transforms3d as t3d
import base64
from typing import Optional, Dict, List

from common.config_loader import config_loader
from common.logger_loader import logger
from roborpc.sim_robots.sim_robot_interface import SimRobotInterface

try:
    while is_stage_loading():
        simulation_app.update()
except (Exception,):
    logger.error('Error in init: %s' % traceback.format_exc())

if World.instance():
    World.instance().clear_instance()


class IsaacSimFrankaApp:
    def __init__(self):
        if World.instance():
            World.instance().clear_instance()
        self.world = World(stage_units_in_meters=1)

        asset_path = "/home/jz08/code_repo/franka_isaac/scenes/123.usd"
        lab_asset_path = "/home/jz08/code_repo/franka_isaac/scenes/lab_colmo_cab_can_1_7_auto_open2.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        # add_reference_to_stage(usd_path=lab_asset_path, prim_path="/World/Lab")

        self._NAMESPACE = config_loader.config['robot']['config']['namespace']
        self._NAMESPACE_ROOT = config_loader.config['robot']['config']['namespace_root']

        self._varied_1_cam_config = config_loader.config['robot']['varied_1_camera']
        self._varied_1_cam_prefix = self._NAMESPACE + '/base_link' + '/varied_1_camera'
        self._varied_1_cam_name = self._varied_1_cam_config['name']
        self._varied_1_cam_offset = Gf.Vec3d(*self._varied_1_cam_config['offset'])
        self._varied_1_cam_orientation = self._varied_1_cam_config['orientation']
        self._varied_1_cam_width = self._varied_1_cam_config['width']
        self._varied_1_cam_height = self._varied_1_cam_config['height']
        self._varied_1_cam_hori_aperture = self._varied_1_cam_config['hori_aperture']
        self._varied_1_cam_vert_aperture = self._varied_1_cam_config['vert_aperture']
        self._varied_1_cam_projection = self._varied_1_cam_config['projection']
        self._varied_1_cam_focal_length = self._varied_1_cam_config['focal_length']
        self._varied_1_cam_focus_distance = self._varied_1_cam_config['focus_distance']
        self._varied_1_cam_ros2_depth_topic = self._varied_1_cam_config['ros2_topic']['depth_topic']
        self._varied_1_cam_ros2_rgb_topic = self._varied_1_cam_config['ros2_topic']['rgb_topic']
        self._varied_1_cam_ros2_info_topic = self._varied_1_cam_config['ros2_topic']['info_topic']

        self._varied_2_cam_config = config_loader.config['robot']['varied_2_camera']
        self._varied_2_cam_prefix = self._NAMESPACE + '/base_link' + '/varied_2_camera'
        self._varied_2_cam_name = self._varied_2_cam_config['name']
        self._varied_2_cam_offset = Gf.Vec3d(*self._varied_2_cam_config['offset'])
        self._varied_2_cam_orientation = self._varied_2_cam_config['orientation']
        self._varied_2_cam_width = self._varied_2_cam_config['width']
        self._varied_2_cam_height = self._varied_2_cam_config['height']
        self._varied_2_cam_hori_aperture = self._varied_2_cam_config['hori_aperture']
        self._varied_2_cam_vert_aperture = self._varied_2_cam_config['vert_aperture']
        self._varied_2_cam_projection = self._varied_2_cam_config['projection']
        self._varied_2_cam_focal_length = self._varied_2_cam_config['focal_length']
        self._varied_2_cam_focus_distance = self._varied_2_cam_config['focus_distance']
        self._varied_2_cam_ros2_depth_topic = self._varied_2_cam_config['ros2_topic']['depth_topic']
        self._varied_2_cam_ros2_rgb_topic = self._varied_2_cam_config['ros2_topic']['rgb_topic']
        self._varied_2_cam_ros2_info_topic = self._varied_2_cam_config['ros2_topic']['info_topic']

        # Prepare handeye-camera configs.
        self._handeye_cam_config = config_loader.config['robot']['handeye_camera']
        self._handeye_cam_prefix = self._NAMESPACE + '/panda_link8' + '/handeye_camera'
        self._handeye_cam_name = self._handeye_cam_config['name']
        self._handeye_cam_name = self._handeye_cam_config['name']
        self._handeye_cam_offset = Gf.Vec3d(*self._handeye_cam_config['offset'])
        self._handeye_cam_orientation = self._handeye_cam_config['orientation']
        self._handeye_cam_width = self._handeye_cam_config['width']
        self._handeye_cam_height = self._handeye_cam_config['height']
        self._handeye_cam_hori_aperture = self._handeye_cam_config['hori_aperture']
        self._handeye_cam_vert_aperture = self._handeye_cam_config['vert_aperture']
        self._handeye_cam_projection = self._handeye_cam_config['projection']
        self._handeye_cam_focal_length = self._handeye_cam_config['focal_length']
        self._handeye_cam_focus_distance = self._handeye_cam_config['focus_distance']
        self._handeye_cam_ros2_depth_topic = self._handeye_cam_config['ros2_topic']['depth_topic']
        self._handeye_cam_ros2_rgb_topic = self._handeye_cam_config['ros2_topic']['rgb_topic']
        self._handeye_cam_ros2_info_topic = self._handeye_cam_config['ros2_topic']['info_topic']


        handeye_cam = Camera(
            prim_path=self._NAMESPACE + "/panda_link8/handeye_camera",
            position=np.array(self._handeye_cam_offset),
            frequency=20,
            resolution=(256, 256),
            orientation=rot_utils.euler_angles_to_quats(np.array(self._handeye_cam_orientation), degrees=True),
        )

        varied_1_cam = Camera(
            prim_path=self._NAMESPACE + "/base_link/varied_1_camera",
            position=np.array(self._varied_1_cam_offset),
            frequency=20,
            resolution=(256, 256),
            orientation=rot_utils.euler_angles_to_quats(np.array(self._varied_1_cam_orientation), degrees=True),
        )

        varied_2_cam = Camera(
            prim_path=self._NAMESPACE + "/base_link/varied_2_camera",
            position=np.array(self._varied_2_cam_offset),
            frequency=20,
            resolution=(256, 256),
            orientation=rot_utils.euler_angles_to_quats(np.array(self._varied_2_cam_orientation), degrees=True),
        )
        self.viewports = {
            "handeye_camera": handeye_cam,
            "varied_1_camera": varied_1_cam,
            "varied_2_camera": varied_2_cam,
        }
        position = np.array([0, 0, 0])
        rpy_radian = np.array([0.0, 0.0, 0.0])
        orientation = t3d.euler.euler2quat(*rpy_radian, axes='sxyz')

        self.robot_franka = omni.isaac.core.robots.Robot(
                prim_path="/World/panda",
                name="panda",
                position=position,
                orientation=orientation,
                scale=np.array([1, 1, 1]),
            )
        self.world.scene.add(self.robot_franka)

        self.varied_1_cam_xform = XFormPrim(prim_path=self._varied_1_cam_prefix)
        self.varied_2_cam_xform = XFormPrim(prim_path=self._varied_2_cam_prefix)
        self.handeye_cam_xform = XFormPrim(prim_path=self._handeye_cam_prefix)
        self._root_prim = XFormPrim(prim_path=self._NAMESPACE)
        self._arm_base_prim = XFormPrim(prim_path=self._NAMESPACE + '/base_link')
        self._tool_link_prim = XFormPrim(prim_path=self._NAMESPACE + '/panda_link8')

        self.last_rgb = np.zeros((self._varied_1_cam_height, self._varied_1_cam_width, 4), dtype=np.float32)
        self.last_depth = np.zeros((self._varied_1_cam_height, self._varied_1_cam_width), dtype=np.float32)

        self._joint_names = [
                    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5",
                    "panda_joint6", "panda_joint7"
                ]
        # self._gripper_names = ["finger_joint", "right_outer_knuckle_joint"]
        self._gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]

        self.robot_dof = len(self._joint_names) + len(self._gripper_names)
        self.arm_dof = len(self._joint_names)
        self.gripper_dof = len(self._gripper_names)

        self.joint_positions = np.array([0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0])
        self.gripper_velocities = np.array([0.0, 0.0])

        robot_rpc = SingleFrankaInterface2023(self)

        def _listener_rpc() -> None:
            try:
                import zerorpc
                s = zerorpc.Server(robot_rpc)
                s.bind("tcp://0.0.0.0:4244")
                s.run()
            except (Exception,):
                logger.error('Error in DaemonLauncher._listener_rpc: %s' % traceback.format_exc())

        threading.Thread(target=_listener_rpc, name='RpcListener', daemon=False).start()

        # self.add_scene_object()
        self.world.add_physics_callback("execute_step_callback_fn", callback_fn=self.execute_step_callback_fn)
        self.run()

    def run(self):
        articulation_controller = None
        i = 0
        while simulation_app.is_running():
            self.world.step(render=True)
            if not self.world.is_playing():
                if i % 100 == 0:
                    print("**** Click Play to start simulation *****")
                i += 1
                continue

            step_index = self.world.current_time_step_index
            if articulation_controller is None:
                articulation_controller = self.robot_franka.get_articulation_controller()
                print(articulation_controller)
            if step_index < 2:
                self.world.reset()
                self.robot_franka._articulation_view.initialize()
                self.arm_idx_list = [self.robot_franka.get_dof_index(x) for x in self._joint_names]
                self.gripper_idx_list = [self.robot_franka.get_dof_index(x) for x in self._gripper_names]
                print(self.arm_idx_list, self.gripper_idx_list)

                self.robot_franka.set_joint_positions(np.array([0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0]), self.arm_idx_list)
                self.robot_franka._articulation_view.set_max_efforts(
                    values=np.array([5000 for i in range(len(self.arm_idx_list))]), joint_indices=self.arm_idx_list
                )
                self.robot_franka.set_joint_positions(np.array([0.04, 0.04]), self.gripper_idx_list)
                self.robot_franka._articulation_view.set_max_efforts(
                    values=np.array([5000 for i in range(len(self.gripper_idx_list))]), joint_indices=self.gripper_idx_list
                )
                # self.robot_franka.set_joint_velocities(np.array([10, 10]), self.gripper_idx_list)
            # self.robot_franka.set_joint_positions(self.joint_positions, self.arm_idx_list)
            # self.robot_franka.set_joint_velocities(self.gripper_velocities, self.gripper_idx_list)
            # art_action = ArticulationAction(
            #     np.array([0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.5]),
            #     joint_indices=idx_list,
            # )
            # articulation_controller.apply_action(art_action)

            # art_action = ArticulationAction(
            #     np.array([130, 130]),
            #     np.array([130, 130]),
            #     joint_indices=gripper_idx_list,
            # )
            # articulation_controller.apply_action(art_action)

        simulation_app.close()

    def execute_step_callback_fn(self, step_size) -> None:
        self.robot_franka.set_joint_positions(self.joint_positions, self.arm_idx_list)
        self.robot_franka.set_joint_positions(self.gripper_velocities, self.gripper_idx_list)
        # self.robot_franka.set_joint_velocities(self.gripper_velocities, self.gripper_idx_list)

    def get_robot_state(self):
        # End-effector world pose.
        end_effector_world_pose = self._tool_link_prim.get_world_pose()
        end_effector_world_position = (end_effector_world_pose[0] / 100).tolist()  # Convert from cm to meter.
        end_effector_world_quaternions = (end_effector_world_pose[1]).tolist()
        end_effector_world_euler = t3d.euler.quat2euler(end_effector_world_quaternions, axes='sxyz')
        end_effector_world_pose_data = [*end_effector_world_position, *end_effector_world_euler]

        joints_radian = (self.robot_franka.get_joint_positions()).tolist()
        joints_velocities = (self.robot_franka.get_joint_velocities()).tolist()
        joint_torques_computed = np.zeros(self.arm_dof).tolist()
        gripper_joints_radian = joints_radian[self.arm_dof:self.robot_dof]

        # Arm joints radian. Correcting arm (joints) states.
        arm_joints_radian = joints_radian[:self.arm_dof]
        arm_joints_velocities = joints_velocities[:self.arm_dof]
        arm_joint_torques_computed = joint_torques_computed[:self.arm_dof]
        end_effector_world_pose = list(end_effector_world_pose_data)

        robot_states = {
            "cartesian_position": end_effector_world_pose,
            "gripper_position": gripper_joints_radian,
            "joint_position": arm_joints_radian,
            "joint_velocities": arm_joints_velocities,
            "joint_torques_computed": arm_joint_torques_computed,
            "prev_joint_torques_computed": arm_joint_torques_computed,
            "prev_joint_torques_computed_safened": arm_joint_torques_computed,
            "motor_torques_measured": arm_joint_torques_computed,
            # "prev_controller_latency_ms": 0.0,
            # "prev_command_successful": True,
        }

        return robot_states

    def get_joint_positions(self):
        joints_radian = (self.robot_franka.get_joint_positions()).tolist()
        arm_joints_radian = joints_radian[:self.arm_dof]
        return arm_joints_radian

    def get_gripper_position(self):
        joints_radian = (self.robot_franka.get_joint_positions()).tolist()
        print(joints_radian)
        gripper_joints_radian = joints_radian[self.arm_dof]
        return gripper_joints_radian

    def get_joint_velocities(self):
        joints_velocities = (self.robot_franka.get_velocities()).tolist()
        arm_joints_velocities = joints_velocities[:self.arm_dof]
        return arm_joints_velocities

    def get_ee_pose(self):
        # End-effector world pose.
        end_effector_world_pose = self._tool_link_prim.get_world_pose()
        end_effector_world_position = end_effector_world_pose[0] / 100  # Convert from cm to meter.
        end_effector_world_quaternions = end_effector_world_pose[1]
        end_effector_world_euler = t3d.euler.quat2euler(end_effector_world_quaternions, axes='sxyz')
        end_effector_world_pose_data = [*end_effector_world_position, *end_effector_world_euler]
        print(end_effector_world_pose_data)
        return np.asarray(end_effector_world_pose_data).tolist()

    def get_camera_intrinsics(self):
        # self.intrinsics = {
        #     "handeye_instrinsics": self.viewports["handeye_camera"].get_intrinsics_matrix().tolist(),
        #     "varied_1_instrinsics": self.viewports["varied_1_camera"].get_intrinsics_matrix().tolist(),
        #     "varied_2_instrinsics": self.viewports["varied_2_camera"].get_intrinsics_matrix().tolist(),
        # }
        self.intrinsics = {
            "handeye_instrinsics": [],
            "varied_1_instrinsics": [],
            "varied_2_instrinsics": [],
        }
        return self.intrinsics

    def get_camera_extrinsics(self):
        handeye_camera_world_pose = self.handeye_cam_xform.get_world_pose()
        handeye_camera_world_position = (handeye_camera_world_pose[0] / 100).tolist()  # Convert from cm to meter.
        handeye_camera_world_quaternions = (
        handeye_camera_world_pose[1]).tolist()  # Quaternion is scalar-first (w, x, y, z).
        handeye_camera_world_euler = t3d.euler.quat2euler(handeye_camera_world_quaternions, axes='sxyz')
        handeye_camera_world_pose_data = [*handeye_camera_world_position, *handeye_camera_world_euler]

        varied_1_camera_world_pose = self.varied_1_cam_xform.get_world_pose()
        varied_1_camera_world_position = (varied_1_camera_world_pose[0] / 100).tolist()  # Convert from cm to meter.
        varied_1_camera_world_quaternions = (
        varied_1_camera_world_pose[1]).tolist()  # Quaternion is scalar-first (w, x, y, z).
        varied_1_camera_world_euler = t3d.euler.quat2euler(varied_1_camera_world_quaternions, axes='sxyz')
        varied_1_camera_world_pose_data = [*varied_1_camera_world_position, *varied_1_camera_world_euler]

        varied_2_camera_world_pose = self.varied_2_cam_xform.get_world_pose()
        varied_2_camera_world_position = (varied_2_camera_world_pose[0] / 100).tolist()  # Convert from cm to meter.
        varied_2_camera_world_quaternions = (
        varied_2_camera_world_pose[1]).tolist()  # Quaternion is scalar-first (w, x, y, z).
        varied_2_camera_world_euler = t3d.euler.quat2euler(varied_2_camera_world_quaternions, axes='sxyz')
        varied_2_camera_world_pose_data = [*varied_2_camera_world_position, *varied_2_camera_world_euler]

        cameras_extrinsics = {
            "handeye_camera_world_pose": handeye_camera_world_pose_data,
            "varied_1_camera_world_pose": varied_1_camera_world_pose_data,
            "varied_2_camera_world_pose": varied_2_camera_world_pose_data,
        }
        return cameras_extrinsics

    def _collect_camera_rgb_data(self, viewport_window):
        try:
            rgb = np.asarray(viewport_window.get_rgba(), dtype=np.float32)
            self.last_rgb = rgb
            return rgb
        except Exception as e:
            print(e)
            return self.last_rgb

    def _collect_camera_depth_data(self, viewport_window):
        try:
            # depth = syn.sensors.get_depth_linear(viewport_window)
            depth = np.asarray(viewport_window.get_depth(), dtype=np.float32)
            # numpy has nan
            depth[np.where(np.isnan(depth))] = 0.0
            self.last_depth = depth
            return depth
        except Exception as e:
            print(e)
            return self.last_depth

    def read_isaac_sim_camera(self):
        # rgb
        handeye_rgb = self._collect_camera_rgb_data(self.viewports["handeye_camera"])[:, :, :3][..., ::-1]
        varied_1_rgb = self._collect_camera_rgb_data(self.viewports["varied_1_camera"])[:, :, :3][..., ::-1]
        varied_2_rgb = self._collect_camera_rgb_data(self.viewports["varied_2_camera"])[:, :, :3][..., ::-1]
        print(handeye_rgb)
        # depth
        handeye_depth = self._collect_camera_depth_data(self.viewports["handeye_camera"])
        print(handeye_depth)
        handeye_depth[np.where(handeye_depth > 6.5)] = 0.0
        handeye_depth = handeye_depth * 10000
        varied_1_depth = self._collect_camera_depth_data(self.viewports["varied_1_camera"])
        varied_1_depth[np.where(varied_1_depth > 6.5)] = 0.0
        varied_1_depth = varied_1_depth * 10000
        varied_2_depth = self._collect_camera_depth_data(self.viewports["varied_1_camera"])
        varied_2_depth[np.where(varied_1_depth > 6.5)] = 0.0
        varied_2_depth = varied_1_depth * 10000
        return {
            "isaac_sim_handeye_rgb": handeye_rgb,
            "isaac_sim_handeye_depth": handeye_depth,
            "isaac_sim_varied_1_rgb": varied_1_rgb,
            "isaac_sim_varied_1_depth": varied_1_depth,
            "isaac_sim_varied_2_rgb": varied_2_rgb,
            "isaac_sim_varied_2_depth": varied_2_depth,
        }

    def init_robot_posture(self, step_num: int, robot_init_arm_joints, robot_init_gripper_degrees,
                           robot_init_body_position) -> None:

        robot_init_arm_joints = [
            0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0,
        ]

        diff_arm = np.zeros(self.arm_dof)

        if robot_init_arm_joints is not None:
            diff_arm = (np.array(robot_init_arm_joints)- np.zeros(self.arm_dof)) / float(step_num)

        self.joint_positions += diff_arm

        for viewport_id, viewport in self.viewports.items():
            viewport.initialize()
            viewport.add_distance_to_image_plane_to_frame()

    def add_scene_object(self):
        cube = self.world.scene.add(
            DynamicCuboid(
                name="cube",
                position=np.array([0.5, 0.5, 0.3]),
                prim_path=self._NAMESPACE_ROOT + "/Cube",
                scale=np.array([0.0515, 0.0515, 0.0715]),
                size=1.0,
                color=np.array([0, 0, 1]),
            )
        )


class SingleFrankaInterface2023(SimRobotInterface):

    def __init__(self, robot):
        self.robot = robot

    def update_command(self, command: List[float], action_apace: str = "cartesian_velocity",
                       gripper_action_space: Optional[str] = "position", blocking: bool = False):
        """Update the command for the robot.

        Args:
            command: Command to be sent to the robot, of shape (12,), in unit m/s for cartesian_velocity, rad/s for joint_velocity, and unit degree for gripper_position.
            action_apace: Action space for the command, either "cartesian_velocity" or "joint_velocity".
            gripper_action_space: Action space for the gripper, either "position" or "velocity".
            blocking: Whether to block until the command is executed.
        """
        assert action_apace == "joint_position", "Only support joint positions control for single Franka robot now."
        assert gripper_action_space == "gripper_position", "Only support position control for gripper for single Franka robot now."
        assert len(command) == 8, "Command should be of length 8."
        self.robot.joint_positions = command[:7]
        self.robot.gripper_velocities = np.concatenate([command[7], command[7]])

    def update_pose(self, command: List[float], velocity: bool = False, blocking: bool = False):
        """Update the pose of the robot."""
        raise NotImplementedError("Do not support pose control for single Franka robot now.")

    def update_joints(self, command: List[float], velocity: bool = False,
                      blocking: bool = False, cartesian_noise: Optional[List[float]] = None, ):
        """Update the joints of the robot."""
        print("Update joints")
        assert len(command) == 7, "Command should be of length 7 for joint positions."
        joints = np.asarray(command)
        self.robot.joint_positions = joints

    def update_gripper(self, command: List[float], velocity: bool = False, blocking: bool = False):
        """Update the gripper of the robot."""
        self.robot.gripper_velocities = np.concatenate([command, command])

    def get_robot_state(self) -> Dict[str, List[float]]:
        """Get the state of the robot.

        Returns:
            A dictionary containing the state of the robot, including "pose", "joints", "gripper_position", "gripper_velocity", "cartesian_velocity", "joint_velocity".
        """
        return self.robot.get_robot_state()

    def get_joint_positions(self) -> List[float]:
        """Get the joint positions of the robot."""
        return self.robot.get_joint_positions()

    def get_gripper_position(self) -> List[float]:
        """Get the gripper position of the robot."""
        return self.robot.get_gripper_position()

    def get_joint_velocities(self) -> List[float]:
        """Get the joint velocity of the robot."""
        return self.robot.get_joint_velocities()

    def get_ee_pose(self) -> List[float]:
        """Get the pose of the robot."""
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

    def read_camera(self):
        """Read a frame from the camera.
        """

        def rgb_to_base64(rgb, size=None, quality=10):
            height, width = rgb.shape[0], rgb.shape[1]
            if size is not None:
                new_height, new_width = size, int(size * float(width) / height)
                rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # webp seems to be better than png and jpg as a codec, in both compression and quality
            encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
            fmt = ".webp"

            _, rgb_data = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_param)
            return base64.b64encode(rgb_data).decode("utf-8")

        def depth_to_base64(depth, size=None, quality=10):
            height, width = depth.shape[0], depth.shape[1]
            if size is not None:
                new_height, new_width = size, int(size * float(width) / height)
                depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            depth_img = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_img = 255 - depth_img

            # webp seems to be better than png and jpg as a codec, in both compression and quality
            encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
            fmt = ".webp"

            _, depth_img_data = cv2.imencode(fmt, depth_img, encode_param)
            return base64.b64encode(depth_img_data).decode("utf-8")

        isaac_sim_camera_dict = self.robot.read_isaac_sim_camera()
        camera_info_dict = {
            "sim_handeye_rgb": rgb_to_base64(isaac_sim_camera_dict["isaac_sim_handeye_rgb"]),
            "sim_varied_1_rgb": rgb_to_base64(isaac_sim_camera_dict["isaac_sim_varied_1_rgb"]),
            "sim_varied_2_rgb": rgb_to_base64(isaac_sim_camera_dict["isaac_sim_varied_2_rgb"]),
            "sim_handeye_depth": depth_to_base64(isaac_sim_camera_dict["isaac_sim_handeye_depth"]),
            "sim_varied_1_depth": depth_to_base64(isaac_sim_camera_dict["isaac_sim_varied_1_depth"]),
            "sim_varied_2_depth": depth_to_base64(isaac_sim_camera_dict["isaac_sim_varied_2_depth"]),
        }
        return camera_info_dict


if __name__ == "__main__":
    IsaacSimFrankaApp()
