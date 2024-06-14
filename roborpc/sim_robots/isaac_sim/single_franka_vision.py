import enum
import math
import os
import pickle
import threading
import time
from typing import Dict, Optional, IO

import cv2
import numpy as np
import numpy.typing as npt
import omni
import omni.isaac.core.utils.rotations as rotations_utils
import omni.graph.core as og
import omni.kit.commands
import omni.kit.viewport_legacy as vp  # Isaac Sim 2022.1.1
import omni.syntheticdata as syn
import omni.usd
import omni.isaac.core.utils.stage as utils_stage
import transforms3d as t3d
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.kit import SimulationApp
from pxr import Gf, UsdGeom
from omni.isaac.urdf import _urdf

from common.config_loader import config_loader
from common.logger_loader import logger
from robots.dual_franka.robot_single_franka import RobotSingleFranka
from robots.robot_interface import RobotInterface


class ControlMode(enum.Enum):
    ARM = 'ARM'
    BASE = 'BASE'
    STOP = 'STOP'


class SingleFrankaVision(RobotInterface):
    _single_lock = threading.Lock()
    _instance = None

    def __new__(cls,
                simulation_app: SimulationApp,
                simulation_context: World,
                robot_usd_path: str,
                init_position: npt.NDArray[np.float64] = np.zeros(3),
                init_orientation: npt.NDArray[np.float64] = np.zeros(3)
                ):
        # single instance
        with cls._single_lock:
            if cls._instance is None:
                print("Creating new instance")
                cls._instance = super().__new__(cls)
                cls._instance.initialize(simulation_app=simulation_app,
                                         simulation_context=simulation_context,
                                         robot_usd_path=robot_usd_path,
                                         init_position=init_position,
                                         init_orientation=init_orientation)
        return cls._instance

    def initialize(
            self,
            simulation_app: SimulationApp,
            simulation_context: World,
            robot_usd_path: str,
            init_position: npt.NDArray[np.float64] = np.zeros(3),
            init_orientation: npt.NDArray[np.float64] = np.zeros(3)
    ) -> None:
        """Robot initialization.

        Set up robot configurations.

        Args:
            simulation_app: SimulationApp.
            robot_usd_path: Robot usd filepath in the directory.
            init_position: Position of the robot, of shape (3,), in unit cm.
            init_orientation: Orientation of the robot, of shape (3,), in unit degree.
        """
        logger.debug("Hello from franka robot vision init")

        self._simulation_context = simulation_context

        # urdf_interface = _urdf.acquire_urdf_interface()
        #
        # import_config = _urdf.ImportConfig()
        # import_config.merge_fixed_joints = False
        # import_config.convex_decomp = False
        # import_config.import_inertia_tensor = True
        # import_config.fix_base = True
        # import_config.make_default_prim = False
        # import_config.self_collision = False
        # import_config.create_physics_scene = True
        # import_config.import_inertia_tensor = False
        # import_config.default_drive_strength = 20000
        # import_config.default_position_drive_damping = 500
        # import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        # import_config.distance_scale = 1
        # import_config.density = 0.0
        #
        # path = str(os.path.abspath(os.path.dirname(__file__)))
        # robot_cfg_path = str(os.path.join(path, 'franka_description'))
        # filename = "franka_panda.urdf"
        #
        # imported_robot = urdf_interface.parse_urdf(robot_cfg_path, filename, import_config)
        # dest_path = ''
        # robot_path = urdf_interface.import_robot(
        #     robot_cfg_path,
        #     filename,
        #     imported_robot,
        #     import_config,
        #     dest_path,
        # )
        #
        robot_name = 'panda',
        self.robot_dof = 9
        self.arm_dof = 7
        self.gripper_dof = 2
        position = np.array([120, -50, 80])
        rpy_radian = np.array([0.0, 0.0, 0.0])
        orientation = t3d.euler.euler2quat(*rpy_radian, axes='sxyz')

        self._simulation_app = simulation_app
        self._usd_path = robot_usd_path
        self._init_position = init_position
        self._init_orientation = init_orientation
        self._NAMESPACE = config_loader.config['robot']['config']['namespace']
        self._NAMESPACE_ROOT = config_loader.config['robot']['config']['namespace_root']

        # # /World/panda
        utils_stage.add_reference_to_stage(self._usd_path, self._NAMESPACE_ROOT)
        self.robot_franka = omni.isaac.core.robots.Robot(
            prim_path=self._NAMESPACE,
            name=robot_name,
            position=position,
            orientation=orientation,
            scale=np.array([100, 100, 100]),
        )

        self._simulation_context.scene.add(self.robot_franka)

        self._joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5",
            "panda_joint6", "panda_joint7", "finger_joint", "right_outer_knuckle_joint",
        ]

        # # Create Redis communication tool.
        # self._redis_pool = redis.ConnectionPool(host=config_loader.config['redis_config']['host'],
        #                                         port=config_loader.config['redis_config']['port'],
        #                                         password=config_loader.config['redis_config']['password'])
        # self._redis = redis.Redis(connection_pool=self._redis_pool)
        # # clear redis key-value
        # redis_keys = self._redis.keys("*")
        # if len(redis_keys) > 0:
        #     self._redis.delete(*self._redis.keys("*"))
        # XFormPrim configs.
        self._root_prim = XFormPrim(prim_path=self._NAMESPACE)

        self._arm_base_prim = XFormPrim(prim_path=self._NAMESPACE + '/base_link')

        self._tool_link_prim = XFormPrim(prim_path=self._NAMESPACE + '/panda_link8')
        # Update robot world pose.
        # self.set_world_pose(self._init_position, self._init_orientation)

        # ----
        self._joint_positions_correction = np.asarray([
            1, 1, 1, 1, 1, 1, 1, 1, 1
        ])
        self._gripper_positions_correction = np.asarray([1, -1])
        # State parameters.
        self.joint_positions = np.zeros(self.robot_dof)  # unit: radian

        self.gripper_positions = np.zeros(self.gripper_dof)  # unit: radian

        # ----
        # Set state and cmd redis keys.
        self._states = config_loader.config['robot_execution']['isaac_sim_gen2']['robot_states']
        self._cmds = config_loader.config['robot_execution']['isaac_sim_gen2']['robot_cmds']
        # Creating redis.
        # self._init_redis_states_cmds()

        # ----
        self._control_mode = ControlMode.STOP
        # Init data collection file.
        self.control_mode_file = None
        self.end_effector_world_pose_file = None
        self.gripper_joints_radian_file = None
        self.arm_joints_radian_file = None

        self.last_control_mode = ''
        self.last_end_effector_world_pose_data = np.zeros(6)
        self.last_gripper_joints_radian = np.zeros(8)
        self.last_arm_joints_radian = np.zeros(7)

        logger.success('Robot gen2 initialized.')

        # Prepare varied-camera configs.
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

        self._stage = omni.usd.get_context().get_stage()

        self.camera_action_graphs = []
        self.create_handeye_cam_action_graph()
        self.create_varied_1_cam_action_graph()
        self.create_varied_2_cam_action_graph()

        for action_graph in self.camera_action_graphs:
            og.Controller.evaluate_sync(action_graph)

        # Create camera viewports, i.e., viewport 0 and 1.
        self.cam_resolutions = [[self._handeye_cam_width, self._handeye_cam_height],
                                [self._varied_1_cam_width, self._varied_1_cam_height],
                                [self._varied_2_cam_width, self._varied_2_cam_height]]
        self.viewport_interface = vp.get_viewport_interface()
        self.viewports = []

        instrinsics_list = ["handeye_instrinsics", "varied_1_instrinsics", "varied_2_instrinsics"]
        self.intrinsics = {}

        for idx, instance in enumerate(self.viewport_interface.get_instance_list()):
            print(f"Viewport {idx}")
            # idx 0 is handeye camera; idx 1 is head camera.
            width, height = self.cam_resolutions[idx]
            viewport = self.viewport_interface.get_viewport_window(instance)
            viewport.set_window_size(width, height)
            viewport.set_texture_resolution(width, height)
            # activate sensor including rgb,depth,bbox
            syn.sensors.create_or_retrieve_sensor(viewport, syn._syntheticdata.SensorType.Rgb)
            syn.sensors.create_or_retrieve_sensor(viewport, syn._syntheticdata.SensorType.DepthLinear)
            syn.sensors.create_or_retrieve_sensor(viewport, syn._syntheticdata.SensorType.BoundingBox2DTight)
            syn.sensors.create_or_retrieve_sensor(viewport, syn._syntheticdata.SensorType.BoundingBox3D)
            syn.sensors.create_or_retrieve_sensor(viewport, syn._syntheticdata.SensorType.SemanticSegmentation)
            self.viewports.append(viewport)

            camera = self._stage.GetPrimAtPath(viewport.get_active_camera())
            focal_length = camera.GetAttribute("focalLength").Get()
            horiz_aperture = camera.GetAttribute("horizontalAperture").Get()
            # Pixels are square so we can do:
            vert_aperture = height / width * horiz_aperture
            near, far = camera.GetAttribute("clippingRange").Get()
            fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
            # compute focal point and center
            focal_x = height * focal_length / vert_aperture
            focal_y = width * focal_length / horiz_aperture
            center_x = height * 0.5
            center_y = width * 0.5
            self.intrinsics[f"{instrinsics_list[idx]}"] = np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]]).tolist()
            # intrinsics["distCoeffs"] = np.array(list(params.disto))


        self.varied_1_cam_xform = XFormPrim(prim_path=self._varied_1_cam_prefix)
        self.varied_2_cam_xform = XFormPrim(prim_path=self._varied_2_cam_prefix)
        self.handeye_cam_xform = XFormPrim(prim_path=self._handeye_cam_prefix)


        self.last_varied_1_camera_world_pose_data = np.zeros(6)
        self.last_varied_2_camera_world_pose_data = np.zeros(6)

        self.handeye_rgb_data_path = None
        self.handeye_depth_data_path = None
        self.varied_1_rgb_data_path = None
        self.varied_1_depth_data_path = None
        self.varied_1_segm_data_path = None

        self.varied_1_camera_world_pose_file = None
        self.last_rgb = np.zeros((self._varied_1_cam_height, self._varied_1_cam_width, 4))
        self.last_depth = np.zeros((self._varied_1_cam_height, self._varied_1_cam_width))

        logger.success('Robot gen2 vision initialized.')

    def _create_cam_prim(
            self,
            prefix: str,
            name: str,
            offset,
            orientation,
            hori_aperture: float,
            vert_aperture: float,
            projection: str,
            focal_length: float,
            focus_distance
    ) -> UsdGeom.Camera:
        camera_path = prefix + name
        camera_prim = UsdGeom.Camera(self._stage.DefinePrim(camera_path, "Camera"))
        xform_api = UsdGeom.XformCommonAPI(camera_prim)
        xform_api.SetTranslate(offset)
        xform_api.SetRotate(orientation, UsdGeom.XformCommonAPI.RotationOrderXYZ)
        camera_prim.GetHorizontalApertureAttr().Set(hori_aperture)
        camera_prim.GetVerticalApertureAttr().Set(vert_aperture)
        camera_prim.GetProjectionAttr().Set(projection)
        camera_prim.GetFocalLengthAttr().Set(focal_length)
        camera_prim.GetFocusDistanceAttr().Set(focus_distance)

        return camera_prim

    def _create_cam_action_graph(
            self,
            viewport_id: int,
            prefix: str,
            name: str,
            rgb_topic: str,
            depth_topic: str,
            info_topic: str,
            node_namespace: str = '',
            is_ros2: bool = True,
    ):
        camera_path = prefix + name

        # ----
        # Check ROS version.
        ros_version = "ROS1"
        ros_bridge_version = "ros_bridge."
        self.ros_vp_offset = 1
        if is_ros2:
            ros_version = "ROS2"
            ros_bridge_version = "ros2_bridge."
            self.ros_vp_offset = 0  # Only create 2 viewports.

        # ----
        # Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers.
        keys = og.Controller.Keys
        (camera_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": "/ROS_" + name.split("/")[-1] + "_Graph",
                "evaluator_name": "push",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                    ("setActiveCamera", "omni.graph.ui.SetActiveViewportCamera"),

                    ("cameraHelperRgb", "omni.isaac." + ros_bridge_version + ros_version + "CameraHelper"),
                    ("cameraHelperDepth", "omni.isaac." + ros_bridge_version + ros_version + "CameraHelper"),
                    ("cameraHelperInfo", "omni.isaac." + ros_bridge_version + ros_version + "CameraHelper"),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    ("createViewport.outputs:execOut", "setActiveCamera.inputs:execIn"),
                    ("createViewport.outputs:viewport", "setActiveCamera.inputs:viewport"),

                    ("setActiveCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    ("setActiveCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    ("setActiveCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),

                    ("createViewport.outputs:viewport", "cameraHelperRgb.inputs:viewport"),
                    ("createViewport.outputs:viewport", "cameraHelperDepth.inputs:viewport"),
                    ("createViewport.outputs:viewport", "cameraHelperInfo.inputs:viewport"),
                ],
                keys.SET_VALUES: [
                    ("createViewport.inputs:viewportId", viewport_id + self.ros_vp_offset),
                    ("setActiveCamera.inputs:primPath", camera_path),

                    ("cameraHelperRgb.inputs:frameId", name),
                    ("cameraHelperRgb.inputs:nodeNamespace", node_namespace),
                    ("cameraHelperRgb.inputs:topicName", rgb_topic),
                    ("cameraHelperRgb.inputs:type", "rgb"),

                    ("cameraHelperDepth.inputs:frameId", name),
                    ("cameraHelperDepth.inputs:nodeNamespace", node_namespace),
                    ("cameraHelperDepth.inputs:topicName", depth_topic),
                    ("cameraHelperDepth.inputs:type", "depth"),

                    ("cameraHelperInfo.inputs:frameId", name),
                    ("cameraHelperInfo.inputs:nodeNamespace", node_namespace),
                    ("cameraHelperInfo.inputs:topicName", info_topic),
                    ("cameraHelperInfo.inputs:type", "camera_info"),
                ],
            },
        )
        return camera_graph

    def create_varied_1_cam_action_graph(self):
        camera_prim = self._create_cam_prim(
            prefix=self._varied_1_cam_prefix,
            name=self._varied_1_cam_name,
            offset=self._varied_1_cam_offset,
            orientation=self._varied_1_cam_orientation,
            hori_aperture=self._varied_1_cam_hori_aperture,
            vert_aperture=self._varied_1_cam_vert_aperture,
            projection=self._varied_1_cam_projection,
            focal_length=self._varied_1_cam_focal_length,
            focus_distance=self._varied_1_cam_focus_distance
        )
        camera_graph = self._create_cam_action_graph(
            viewport_id=1,
            prefix=self._varied_1_cam_prefix,
            name=self._varied_1_cam_name,
            rgb_topic=self._varied_1_cam_ros2_rgb_topic,
            depth_topic=self._varied_1_cam_ros2_depth_topic,
            info_topic=self._varied_1_cam_ros2_info_topic
        )

        self.camera_action_graphs.append(camera_graph)

    def create_varied_2_cam_action_graph(self):
        camera_prim = self._create_cam_prim(
            prefix=self._varied_2_cam_prefix,
            name=self._varied_2_cam_name,
            offset=self._varied_2_cam_offset,
            orientation=self._varied_2_cam_orientation,
            hori_aperture=self._varied_2_cam_hori_aperture,
            vert_aperture=self._varied_2_cam_vert_aperture,
            projection=self._varied_2_cam_projection,
            focal_length=self._varied_2_cam_focal_length,
            focus_distance=self._varied_2_cam_focus_distance
        )
        camera_graph = self._create_cam_action_graph(
            viewport_id=2,
            prefix=self._varied_2_cam_prefix,
            name=self._varied_2_cam_name,
            rgb_topic=self._varied_2_cam_ros2_rgb_topic,
            depth_topic=self._varied_2_cam_ros2_depth_topic,
            info_topic=self._varied_2_cam_ros2_info_topic
        )

        self.camera_action_graphs.append(camera_graph)

    def create_handeye_cam_action_graph(self):
        camera_prim = self._create_cam_prim(
            prefix=self._handeye_cam_prefix,
            name=self._handeye_cam_name,
            offset=self._handeye_cam_offset,
            orientation=self._handeye_cam_orientation,
            hori_aperture=self._handeye_cam_hori_aperture,
            vert_aperture=self._handeye_cam_vert_aperture,
            projection=self._handeye_cam_projection,
            focal_length=self._handeye_cam_focal_length,
            focus_distance=self._handeye_cam_focus_distance
        )
        camera_graph = self._create_cam_action_graph(
            viewport_id=0,
            prefix=self._handeye_cam_prefix,
            name=self._handeye_cam_name,
            rgb_topic=self._handeye_cam_ros2_rgb_topic,
            depth_topic=self._handeye_cam_ros2_depth_topic,
            info_topic=self._handeye_cam_ros2_info_topic
        )

        self.camera_action_graphs.append(camera_graph)

    def _open_camera_collection_files(self, collect_data_path_prefix: str) -> None:
        self.last_varied_1_camera_world_pose_data = np.zeros(6)

        self.handeye_rgb_data_path = collect_data_path_prefix + '/handeye/rgb'
        self.handeye_depth_data_path = collect_data_path_prefix + '/handeye/depth'
        self.varied_1_rgb_data_path = collect_data_path_prefix + '/varied_1/rgb'
        self.varied_1_depth_data_path = collect_data_path_prefix + '/varied_1/depth'
        self.varied_1_segm_data_path = collect_data_path_prefix + '/varied_1/segm'

        for camera_path in [self.handeye_rgb_data_path,
                            self.handeye_depth_data_path,
                            self.varied_1_rgb_data_path,
                            self.varied_1_depth_data_path,
                            self.varied_1_segm_data_path]:
            if not os.path.exists(camera_path):
                os.makedirs(camera_path)

        varied_1_camera_world_pose_file_path = os.path.join(collect_data_path_prefix, 'varied_1_camera_world_pose.txt')
        self.varied_1_camera_world_pose_file = open(varied_1_camera_world_pose_file_path, 'a+')

    def _close_camera_collection_files(self) -> None:
        if self.varied_1_camera_world_pose_file is not None:
            self.varied_1_camera_world_pose_file.close()
            self.varied_1_camera_world_pose_file = None

    def flush_camera_states_collection_files(self) -> None:
        self.varied_1_camera_world_pose_file.flush()

    def check_obj_in_fov(self, obj_prim_path):
        """ Data shape of bounding 2D
            uniqueId,               e.g. 211
            name,                   e.g. /Dynamic_Items/Shoe_1
            semanticLabel           e.g. Shoe !
            metadata                e.g. ''
            instanceIds             e.g. List[int]
            semanticId              e.g.  26
            x_min                   in pixel
            y_min
            x_max
            y_max
            Returns : tuple with the above described shape
        """
        # return True
        logger.debug('flag!!!!!')
        bbox_2d_list = syn.sensors.get_bounding_box_2d_tight(self.viewports[1])  # TODO BUG
        logger.debug(f'bbox_2d_list: {bbox_2d_list}')
        if len(bbox_2d_list) < 1:
            return False
        for line in bbox_2d_list:
            prim_str = line[1]
            # print("prim_str", prim_str)
            # print("obj_prim_path",obj_prim_path)
            if prim_str == obj_prim_path:
                return True
        return False

    def _collect_camera_states(self) -> None:
        # Head camera world pose.
        # The camera world pose will only change with the movement of the varied_1.
        varied_1_camera_world_pose = self.varied_1_cam_xform.get_world_pose()
        varied_1_camera_world_position = varied_1_camera_world_pose[0] / 100  # Convert from cm to meter.
        varied_1_camera_world_quaternions = varied_1_camera_world_pose[1]  # Quaternion is scalar-first (w, x, y, z).
        varied_1_camera_world_euler = t3d.euler.quat2euler(varied_1_camera_world_quaternions, axes='sxyz')
        varied_1_camera_world_pose_data = [*varied_1_camera_world_position, *varied_1_camera_world_euler]

        self.varied_1_camera_world_pose_file.write(str(varied_1_camera_world_pose_data) + '\n')
        self.last_varied_1_camera_world_pose_data = varied_1_camera_world_pose_data

        self.flush_camera_states_collection_files()

    def _collect_camera_data(self, file_prefix: str, label2id_dict: Optional[Dict[str, int]] = None) -> None:
        """Collect camera pictures."""
        # rgb
        handeye_rgb = self._collect_camera_rgb_data(self.viewports[0])[:, :, :3][..., ::-1]
        self.save_camera_rgb_data(self.handeye_rgb_data_path, file_prefix, handeye_rgb)
        varied_1_rgb = self._collect_camera_rgb_data(self.viewports[1])[:, :, :3][..., ::-1]
        self.save_camera_rgb_data(self.varied_1_rgb_data_path, file_prefix, varied_1_rgb)

        # depth
        handeye_depth = self._collect_camera_depth_data(self.viewports[0])
        handeye_depth[np.where(handeye_depth > 6.5)] = 0.0
        handeye_depth = handeye_depth * 10000
        handeye_depth_data = handeye_depth.astype(np.uint16)
        self.save_camera_depth_data(self.handeye_depth_data_path, file_prefix, handeye_depth_data)
        varied_1_depth = self._collect_camera_depth_data(self.viewports[1])
        varied_1_depth[np.where(varied_1_depth > 6.5)] = 0.0
        varied_1_depth = varied_1_depth * 10000
        varied_1_depth_data = varied_1_depth.astype(np.uint16)
        self.save_camera_depth_data(self.varied_1_depth_data_path, file_prefix, varied_1_depth_data)

        if label2id_dict is not None:
            # 2d bbox
            # handeye_bbox_2d = self._collect_camera_bbox_2d(self.viewports[0],label2id_dict)
            # self.save_camera_bbox_2d(segm_data_path,file_prefix,handeye_bbox_2d)
            varied_1_bbox_2d = self._collect_camera_bbox_2d(self.viewports[1], label2id_dict)
            self.save_camera_bbox_2d(self.varied_1_segm_data_path, file_prefix, varied_1_bbox_2d)

        # instance mapping
        # handeye_instance_mapping, handeye_segmentation_data = self._collect_camera_instance_mapping_seg_data(self.viewports[0])
        # self.save_camera_instance_mapping(path_prefix,file_prefix,handeye_instance_mapping)
        # self.save_camera_segmentation_data(path_prefix,file_prefix,handeye_segmentation_data)
        varied_1_instance_mapping, varied_1_segmentation_data = self._collect_camera_instance_mapping_seg_data(
            self.viewports[1])
        self.save_camera_instance_mapping(self.varied_1_segm_data_path, file_prefix, varied_1_instance_mapping)
        self.save_camera_segmentation_data(self.varied_1_segm_data_path, file_prefix, varied_1_segmentation_data)

    def get_camera_intrinsics(self):
        return self.intrinsics

    def get_camera_extrinsics(self):
        handeye_camera_world_pose = self.handeye_cam_xform.get_world_pose()
        handeye_camera_world_position = (handeye_camera_world_pose[0] / 100).tolist()  # Convert from cm to meter.
        handeye_camera_world_quaternions = (handeye_camera_world_pose[1]).tolist()  # Quaternion is scalar-first (w, x, y, z).
        handeye_camera_world_euler = t3d.euler.quat2euler(handeye_camera_world_quaternions, axes='sxyz')
        handeye_camera_world_pose_data = [*handeye_camera_world_position, *handeye_camera_world_euler]

        varied_1_camera_world_pose = self.varied_1_cam_xform.get_world_pose()
        varied_1_camera_world_position = (varied_1_camera_world_pose[0] / 100).tolist()  # Convert from cm to meter.
        varied_1_camera_world_quaternions = (varied_1_camera_world_pose[1]).tolist()  # Quaternion is scalar-first (w, x, y, z).
        varied_1_camera_world_euler = t3d.euler.quat2euler(varied_1_camera_world_quaternions, axes='sxyz')
        varied_1_camera_world_pose_data = [*varied_1_camera_world_position, *varied_1_camera_world_euler]

        varied_2_camera_world_pose = self.varied_2_cam_xform.get_world_pose()
        varied_2_camera_world_position = (varied_2_camera_world_pose[0] / 100).tolist()  # Convert from cm to meter.
        varied_2_camera_world_quaternions = (varied_2_camera_world_pose[1]).tolist()  # Quaternion is scalar-first (w, x, y, z).
        varied_2_camera_world_euler = t3d.euler.quat2euler(varied_2_camera_world_quaternions, axes='sxyz')
        varied_2_camera_world_pose_data = [*varied_2_camera_world_position, *varied_2_camera_world_euler]

        cameras_extrinsics = {
            "handeye_camera_world_pose": handeye_camera_world_pose_data,
            "varied_1_camera_world_pose": varied_1_camera_world_pose_data,
            "varied_2_camera_world_pose": varied_2_camera_world_pose_data,
        }
        return cameras_extrinsics


    def read_isaac_sim_camera(self):
        # rgb
        handeye_rgb = self._collect_camera_rgb_data(self.viewports[0])[:, :, :3][..., ::-1]
        varied_1_rgb = self._collect_camera_rgb_data(self.viewports[1])[:, :, :3][..., ::-1]
        varied_2_rgb = self._collect_camera_rgb_data(self.viewports[2])[:, :, :3][..., ::-1]

        # depth
        handeye_depth = self._collect_camera_depth_data(self.viewports[0])
        handeye_depth[np.where(handeye_depth > 6.5)] = 0.0
        handeye_depth = handeye_depth * 10000
        varied_1_depth = self._collect_camera_depth_data(self.viewports[1])
        varied_1_depth[np.where(varied_1_depth > 6.5)] = 0.0
        varied_1_depth = varied_1_depth * 10000
        varied_2_depth = self._collect_camera_depth_data(self.viewports[1])
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

    def _collect_camera_rgb_data(self, viewport_window):
        """ get rgba data from the current camera
        Params
        ======
        None
        Returns: np.ndarray of shape(heigth, width, 4)
        """
        try:
            rgb = syn.sensors.get_rgb(viewport_window)
            self.last_rgb = rgb
            return rgb
        except:
            return self.last_rgb

        # rgb_data = (rgb[:, :, :3])[..., ::-1]
        # cv2.imwrite(os.path.join(path_prefix, f'rgb_cv2_{file_prefix}.png'), rgb_data)

    def _collect_camera_depth_data(self, viewport_window):
        """ get depth data from the current camera
        Params
        ======
        None

        Returns: np.ndarray(height X width) in mm
        """
        try:
            depth = syn.sensors.get_depth_linear(viewport_window)
            self.last_depth = depth
            return depth
        except:
            return self.last_depth
        # depth[np.where(depth > 6.5)] = 0.0
        # depth = depth * 10000
        # depth_data = depth.astype(np.uint16)
        # np.savez_compressed(os.path.join(path_prefix, f'depth_np_{file_prefix}.npy'), depth_data)

    @staticmethod
    def _collect_camera_bbox_2d(viewport_window, label2id_dict: Dict[str, int]):
        bbox_2d_list = syn.sensors.get_bounding_box_2d_tight(viewport_window)
        bbox_res_list = []
        for line in bbox_2d_list:
            label_str = line[2]
            if label_str in label2id_dict:
                res = {}
                x1 = line[-4]
                y1 = line[-3]
                x2 = line[-2]
                y2 = line[-1]
                x_ = (x1 + x2) / 2.0
                y_ = (y1 + y2) / 2.0
                w_ = x2 - x1
                h_ = y2 - y1
                tmp_list = [x_, y_, w_, h_]
                res["2d_bbox"] = tmp_list
                res["prim_path"] = line[1]
                res["category_id"] = label2id_dict[label_str]
                bbox_res_list.append(res)
        return bbox_res_list

    @staticmethod
    def _collect_camera_instance_mapping_seg_data(viewport_window):
        instance_mapping = syn.helpers.get_instance_mappings()
        segmentation_data = syn.sensors.get_instance_segmentation(
            viewport_window, instance_mappings=instance_mapping, return_mapping=False)
        return instance_mapping, segmentation_data

    def _collect_camera_segmentation_map(self, viewport_window, object_prim_path):
        instance_mapping, segmentation_data = self._collect_camera_instance_mapping_seg_data(viewport_window)
        segmentation_data = segmentation_data.astype(np.uint8)
        mask = np.asarray([item[4] for item in instance_mapping
                           if item[1] == object_prim_path]).flatten().astype(np.uint8)
        segmentation_data[np.where(np.isin(segmentation_data, mask))] = 1
        segmentation_data[np.where(segmentation_data != 1)] = 0

        return segmentation_data.astype(np.bool_)

    def save_camera_rgb_data(self, path_prefix: str, file_prefix: str, rgb_data):
        cv2.imwrite(os.path.join(path_prefix, f'rgb_cv2_{file_prefix}.png'), rgb_data)
        # self._collect_camera_rgb_data(viewport_window=self.viewports[0], path_prefix=path_prefix, file_prefix=file_prefix)

    def save_camera_depth_data(self, path_prefix: str, file_prefix: str, depth_data):
        np.savez_compressed(os.path.join(path_prefix, f'depth_np_{file_prefix}.npy'), depth_data)
        # self._collect_camera_depth_data(viewport_window=self.viewports[0], path_prefix=path_prefix, file_prefix=file_prefix)

    def save_camera_bbox_2d(self, path_prefix: str, file_prefix: str, bbox_res_list):
        with open(os.path.join(path_prefix, f'varied_1_bbox_2d_{file_prefix}.pickle'), 'wb') as handle:
            pickle.dump(bbox_res_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_camera_segmentation_data(self, path_prefix: str, file_prefix: str, segmentation_data):
        np.savez_compressed(os.path.join(path_prefix, f'segmentation_data_{file_prefix}.npz'), segmentation_data)

    def save_camera_instance_mapping(self, path_prefix: str, file_prefix: str, instance_mapping):
        with open(os.path.join(path_prefix, f'instance_mapping_{file_prefix}.pickle'), 'wb') as handle:
            pickle.dump(instance_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def open_collection_files(self, collect_data_path_prefix: str) -> None:
        self._open_robot_states_collection_files(collect_data_path_prefix)
        self._open_camera_collection_files(collect_data_path_prefix)

    def close_collection_files(self) -> None:
        self._close_robot_states_collection_files()
        self._close_camera_collection_files()

    def collect_data(self, file_prefix: str, label2id_dict: Optional[Dict[str, int]] = None) -> None:
        robot_states_collect = self._collect_robot_states()
        if robot_states_collect:
            self._collect_camera_states()
            self._collect_camera_data(file_prefix, label2id_dict)

    @property
    def usd_path(self) -> str:
        return self._usd_path

    @property
    def states(self) -> Dict[str, str]:
        return self._states

    @property
    def cmds(self) -> Dict[str, str]:
        return self._cmds

    @property
    def root_prim(self) -> XFormPrim:
        return self._root_prim

    @property
    def body_prim(self) -> XFormPrim:
        return self._root_prim

    @property
    def arm_base_prim(self) -> XFormPrim:
        return self._arm_base_prim

    @property
    def tool_link_prim(self) -> XFormPrim:
        return self._tool_link_prim

    def _init_redis_states_cmds(self) -> None:
        """redis states and cmds initialization."""
        states_var_list = [
            self._states['arm_joints_radian'],
            self._states['gripper_joints_radian']]
        for states_var in states_var_list:
            self._redis.delete(states_var)

        cmds_var_list = [
            self._cmds['arm_joints_radian'],
            self._cmds['gripper_joints_radian']]
        for cmds_var in cmds_var_list:
            self._redis.delete(cmds_var)

        logger.success('Redis states and cmds initialized.')

    def _pub_states_to_redis(self) -> None:
        """Publishing robot states to redis."""
        # Correcting the robot arm (joints) and gripper (joints) states.
        arm_joints_radian = self.joint_positions * self._joint_positions_correction

        # Write robot states to redis variables.
        self._redis.set(self._states['arm_joints_radian'],
                        arm_joints_radian[0:self.robot_dof - self.gripper_dof].tobytes())
        self._redis.set(self._states['gripper_joints_radian'],
                        (arm_joints_radian[self.robot_dof - self.gripper_dof:self.robot_dof] / 100).tobytes())

    def _sub_cmds_from_redis(self) -> bool:
        """Subscribing control cmds from redis list to the corresponding robot joints."""
        set_new_state = False
        self._control_mode = ControlMode.STOP

        # Update arm joints state.
        left_redis_key = self._cmds['arm_joints_radian']
        left_redis_list_len = self._redis.llen(left_redis_key)
        if left_redis_list_len > 0:
            self._control_mode = ControlMode.ARM
            set_new_state = True
            redis_value = self._redis.rpop(left_redis_key)
            joint_new_positions = np.frombuffer(redis_value, dtype=float)

            self.joint_positions[
            0:self.robot_dof - self.gripper_dof] = joint_new_positions * self._joint_positions_correction[
                                                                         0:self.robot_dof - self.gripper_dof]

        # right_redis_key = self._cmds['arm_joints_radian']
        # right_redis_list_len = self._redis.llen(right_redis_key)
        # if right_redis_list_len > 0:
        #     self._control_mode = ControlMode.ARM
        #     set_new_state = True
        #     redis_value = self._redis.rpop(right_redis_key)
        #     joint_new_positions = np.frombuffer(redis_value, dtype=float)
        #
        #     self.joint_positions[9:16] = joint_new_positions * self._joint_positions_correction[9:16]

        # Update gripper joints state.
        left_gripper_redis_key = self._cmds['gripper_joints_radian']
        left_gripper__redis_list_len = self._redis.llen(left_gripper_redis_key)
        if left_gripper__redis_list_len > 0:
            self._control_mode = ControlMode.ARM
            set_new_state = True
            redis_value = self._redis.rpop(left_gripper_redis_key)
            gripper_positions = np.frombuffer(redis_value, dtype=float)

            self.joint_positions[self.robot_dof - self.gripper_dof:self.robot_dof] = gripper_positions

        # right_gripper_redis_key = self._cmds['gripper_joints_radian'] + '_RIGHT'
        # right_gripper__redis_list_len = self._redis.llen(right_gripper_redis_key)
        # if right_gripper__redis_list_len > 0:
        #     self._control_mode = ControlMode.ARM
        #     set_new_state = True
        #     redis_value = self._redis.rpop(right_gripper_redis_key)
        #     gripper_positions = np.frombuffer(redis_value, dtype=float)
        #
        #     self.joint_positions[16:18] = gripper_positions * 100  # unit: cm

        return set_new_state

    def _open_robot_states_collection_files(self, collect_data_path_prefix: str) -> None:
        """Open robot states collection files based on the given file path.

        Returns:
            The flag of file opening.
        """
        self.last_control_mode = ''
        self.last_end_effector_world_pose_data = np.zeros(6)
        self.last_gripper_joints_radian = np.zeros(8)
        self.last_arm_joints_radian = np.zeros(7)

        def open_robot_states_collection_file(path_prefix: str, file_name: str) -> IO:
            file_path = os.path.join(path_prefix, file_name)
            file = open(file_path, 'a+')
            return file

        self.control_mode_file = open_robot_states_collection_file(collect_data_path_prefix, 'control_mode.txt')
        self.end_effector_world_pose_file = open_robot_states_collection_file(collect_data_path_prefix,
                                                                              'end_effector_world_pose.txt')
        self.gripper_joints_radian_file = open_robot_states_collection_file(collect_data_path_prefix,
                                                                            'gripper_joints_radian.txt')
        self.arm_joints_radian_file = open_robot_states_collection_file(collect_data_path_prefix,
                                                                        'arm_joints_radian.txt')

    def _close_robot_states_collection_files(self) -> None:
        def close_robot_states_collection_file(data_file: Optional[IO]):
            if data_file is not None:
                data_file.close()

        files = [self.control_mode_file,
                 self.end_effector_world_pose_file,
                 self.gripper_joints_radian_file,
                 self.arm_joints_radian_file]
        for file in files:
            close_robot_states_collection_file(file)
            file = None

        self.collect_data_file_open_flag = False

    def _flush_robot_states_collection_files(self) -> None:
        files = [self.control_mode_file,
                 self.end_effector_world_pose_file,
                 self.gripper_joints_radian_file,
                 self.arm_joints_radian_file]
        for file in files:
            file.flush()

    def _collect_robot_states(self) -> bool:
        """Collect robot states.

        Returns:
            If the data is collected (i.e., if the robot state is changed).
        """
        # Control mode.
        control_mode: str = self._control_mode.value

        # End-effector world pose.
        end_effector_world_pose = self._tool_link_prim.get_world_pose()
        end_effector_world_position = end_effector_world_pose[0] / 100  # Convert from cm to meter.
        end_effector_world_quaternions = end_effector_world_pose[1]
        end_effector_world_euler = t3d.euler.quat2euler(end_effector_world_quaternions, axes='sxyz')
        end_effector_world_pose_data = [*end_effector_world_position, *end_effector_world_euler]

        # Gripper opening amount.
        gripper_joints_radian = (self.gripper_positions * self._gripper_positions_correction).tolist()

        # Arm joints radian. Correcting arm (joints) states.
        arm_joints_radian = (self.joint_positions * self._joint_positions_correction).tolist()

        # only record the significant changes
        if not (self.last_control_mode == control_mode
                and np.allclose(self.last_end_effector_world_pose_data, end_effector_world_pose_data, atol=5e-4)
                and np.allclose(self.last_gripper_joints_radian, gripper_joints_radian, atol=1e-5)
                and np.allclose(self.last_arm_joints_radian, arm_joints_radian, atol=1e-5)):
            self.control_mode_file.write(control_mode + '\n')
            self.end_effector_world_pose_file.write(str(end_effector_world_pose_data) + '\n')
            self.gripper_joints_radian_file.write(str(gripper_joints_radian) + '\n')
            self.arm_joints_radian_file.write(str(arm_joints_radian) + '\n')

            self.last_control_mode = control_mode
            self.last_end_effector_world_pose_data = end_effector_world_pose_data
            self.last_gripper_joints_radian = gripper_joints_radian
            self.last_arm_joints_radian = arm_joints_radian

            self._flush_robot_states_collection_files()

            return True
        return False

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
            "joint_positions": arm_joints_radian,
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
        gripper_joints_radian = joints_radian[-1]
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
        return end_effector_world_pose_data

    def set_world_pose(self, position: npt.NDArray[np.float64], orientation: npt.NDArray[np.float64]) -> None:
        self.root_prim.set_world_pose(position, rotations_utils.euler_angles_to_quat(orientation, True))
        logger.success(f'Set robot world pose to position {position.tolist()} and orientation {orientation.tolist()}.')

    def pub_states_callback_fn(self, step_size) -> None:
        self._pub_states_to_redis()

    def sub_cmds(self) -> bool:
        set_new_state = self._sub_cmds_from_redis()
        return set_new_state

    def execute_step_callback_fn(self, step_size) -> None:
        self.robot_franka._articulation_view.initialize()
        idx_list = [self.robot_franka.get_dof_index(x) for x in self._joint_names]
        self.robot_franka._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(idx_list))]),
            joint_indices=idx_list)
        # if len(self.joint_positions) == self.robot_dof - 1:
        #     new_joint_positions = np.concatenate([self.joint_positions, [self.joint_positions[-1]]])
        # elif len(self.joint_positions) == self.robot_dof - 2:
        #     raise ValueError
        # else:
        #     new_joint_positions = self.joint_positions
        # print(f"self.joint_positions: {self.joint_positions}")
        self.robot_franka.set_joint_positions(self.joint_positions, idx_list)

    def init_robot_posture(self, step_num: int, robot_init_arm_joints, robot_init_gripper_degrees,
                           robot_init_body_position) -> None:

        robot_init_arm_joints = [
            0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0, 0, 0
        ]

        diff_arm = np.zeros(self.robot_dof)

        if robot_init_arm_joints is not None:
            diff_arm = (np.array(robot_init_arm_joints) *
                        self._joint_positions_correction - np.zeros(self.robot_dof)) / float(step_num)

        self.joint_positions += diff_arm