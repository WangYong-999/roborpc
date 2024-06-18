

from omni.isaac.kit import SimulationApp
CONFIG = {"width": 1280, "height": 720, "sync_loads": False,
          "headless": False, "renderer": "RayTracedLighting"}
simulation_app = SimulationApp(CONFIG)  # we can also run as headless.

import zerorpc
import tqdm
import http
import http.server
import json
import numpy as np
np.set_printoptions(precision=4)
import threading
import traceback
from datetime import datetime
from pathlib import Path
from socketserver import ThreadingMixIn

from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config

import omni
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import is_stage_loading
import omni.kit.commands
import omni.kit.primitive.mesh
from omni.physx.scripts.physicsUtils import *
enable_extension("omni.isaac.ros2_bridge")

import carb
import omni.graph.core as og
import omni.kit.commands
import omni.kit.primitive.mesh
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core import SimulationContext, World

try:
    while is_stage_loading():
        simulation_app.update()
except (Exception,):
    logger.error('Error in init: %s' % traceback.format_exc())

from typing import List, Optional

from roborpc.sim_robots.isaac_sim.single_franka_vision import RobotSingleFrankaVision as Robot
from roborpc.sim_robots.sim_robot_interface import SimRobotInterface as RpcRobot

_TIMESTAMP_STRING_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
_LOAD_TASK_EVENT: threading.Event = threading.Event()
_LOADED_TASK_EVENT: threading.Event = threading.Event()


class RobotTask:
    def __init__(self, scene_task_id: str, scene_usd: str, scene_task_name: str,
                 source_object_prim_path: str, target_object_prim_path: str,
                 robot_init_pose: List, command: str, robot_init_arm_joints: Optional[List[float]],
                 robot_init_gripper_degrees: Optional[List[float]], robot_init_body_position: Optional[List[float]],
                 init_object_info:Optional[str]) -> None:
        self.scene_task_id = scene_task_id
        self.scene_usd = scene_usd
        self.environment_path = scene_usd
        self.robot_init_position = robot_init_pose[:3]
        self.robot_init_orientation = robot_init_pose[3:]
        self.scene_task_name = scene_task_name
        self.source_object_prim_path = source_object_prim_path
        self.target_object_prim_path = target_object_prim_path
        self.robot_init_pose = robot_init_pose
        self.command = command
        self.robot_init_arm_joints = robot_init_arm_joints
        self.robot_init_gripper_degrees = robot_init_gripper_degrees
        self.robot_init_body_position = robot_init_body_position
        self.init_object_info = init_object_info


class IsaacSimTaskRunner:
    def __init__(self,
                 simulation_app,
                 robot_task,
                 physics_dt: float = 1.0 / 100,
                 render_dt: float = 1.0 / 100,
                 stage_units_in_meters: float = 0.01,
                 ):
        """Initiate the simulation.

        Args:
            physics_dt: Physics downtime of the scene.
            render_dt: Render downtime of the scene.
            stage_units_in_meters: The state unit.
        """
        self.simulation_app = simulation_app
        self.isaac_config = config['roborpc']['sim_robots']['isaac_sim']
        self.skip_rendering = self.isaac_config['skip_rendering']
        logger.debug(robot_task.environment_path)
        if robot_task.environment_path is None:
            self.environment_path = self.isaac_config['env_usd_path']
        else:
            self.environment_path = robot_task.environment_path

        self.robot_path = self.isaac_config['robot_usd_path']
        if robot_task.robot_init_position is None:  # dont use == None
            self.robot_init_position = np.asarray(self.isaac_config['init_position'])
        else:
            self.robot_init_position = np.asarray(robot_task.robot_init_position)
        if robot_task.robot_init_orientation is None:
            self.robot_init_orientation = np.asarray(self.isaac_config['init_orientation'])
        else:
            self.robot_init_orientation = np.asarray(robot_task.robot_init_orientation)

        # Setting renderer option.
        if self.skip_rendering:
            self._settings = carb.settings.get_settings()
            self._settings.set("/app/renderer/skipMaterialLoading", True)
            # Smaller number reduces material resolution to avoid out-of-memory, max is 15
            self._settings.set("/rtx-transient/resourcemanager/maxMipCount", 0)

        # Load environment.
        omni.usd.get_context().open_stage(self.environment_path)
        omni.usd.get_context().disable_save_to_recent_files()

        self.simulation_context = World(physics_dt=physics_dt,
                                        rendering_dt=render_dt,
                                        stage_units_in_meters=stage_units_in_meters)

        """
        # Load environment objects.
        self._env_objects = config['environment']['env_objects']
        if len(self._env_objects):
            for env_object in self._env_objects:
                object_name = env_object['name']
                object_prefix = env_object['prefix']
                object_usd_path = env_object['usd_path']
                object_position = env_object['position']
                object_orientation = env_object['orientation']

                stage_utils.add_reference_to_stage(object_usd_path, object_prefix)
                XFormPrim(prim_path=object_prefix,
                          name=object_name,
                          position=object_position,
                          orientation=object_orientation)

                while is_stage_loading():
                    simulation_app.update()
        """
        self._robot = Robot(simulation_app=simulation_app, simulation_context=self.simulation_context,
                            robot_usd_path=self.robot_path,
                            init_position=self.robot_init_position,
                            init_orientation=self.robot_init_orientation)

        robot_rpc = RpcRobot(robot=self._robot)

        def _listener_rpc() -> None:
            try:
                s = zerorpc.Server(robot_rpc)
                s.bind(f"tcp://0.0.0.0:{self.isaac_config['rpc_port']}")
                s.run()
            except (Exception,):
                logger.error('Error in DaemonLauncher._listener_rpc: %s' % traceback.format_exc())

        threading.Thread(target=_listener_rpc, name='RpcListener', daemon=False).start()

        # Setting physics option.
        self.simulation_context.get_physics_context().set_broadphase_type("MBP")
        self.simulation_context.get_physics_context().set_solver_type("PGS")
        self.simulation_context.get_physics_context().enable_gpu_dynamics(False)

        while is_stage_loading():
            simulation_app.update()

        # physx_interface = omni.physx.get_physx_interface()
        # physx_interface.overwrite_gpu_setting(1)

        # Init ROS 2 clock.
        self._set_ros2_clock()

        self.set_viewports()
        # self.dock_viewports()

        self.physics_callback_dict = {'execute_step_callback_fn': self._robot.execute_step_callback_fn,
                                      # 'pub_states_callback_fn': self._robot.pub_states_callback_fn,
                                      'evaluate_sync_ros2_clock_graph': self.evaluate_sync_ros2_clock_graph
                                      }

        # Init sim parameters.
        self.sim_time = 0.
        self.sim_step = 0

        logger.success('Runner base initialized.')

        try:
            robot_init_pose = np.array(robot_task.robot_init_pose)
            self._stage = omni.usd.get_context().get_stage()

            self.name_of_scene = None
            self.scene_dir = str(Path(robot_task.scene_usd).parent)
            self.timestamp = None
            self.task_id = None
            self.task_name = None
            self.scene_task_name = robot_task.scene_task_name
            self.source_object_prim_path = robot_task.source_object_prim_path
            self.target_object_prim_path = robot_task.target_object_prim_path
            self.robot_init_arm_joints = robot_task.robot_init_arm_joints
            self.robot_init_gripper_degrees = robot_task.robot_init_gripper_degrees
            self.robot_init_body_position = robot_task.robot_init_body_position

            self.scene_id = None
            self.reset_signal = False

            if robot_task.init_object_info is not None:
                init_objects = json.loads(robot_task.init_object_info)
                for init_object in init_objects:
                    can_xform = XFormPrim(prim_path=init_object['prim_path'], scale=np.array(init_object['scale']))
                    can_xform.set_world_pose(position=np.array(init_object['position']))

            self.contact_report_dict = {}

            while is_stage_loading():
                simulation_app.update()

            logger.success("task runner initialization is Done!")
        except (Exception,):
            logger.error('Error in task runner initialization: %s' % traceback.format_exc())

    def _set_ros2_clock(self) -> None:
        # Creating an ondemand push graph with ROS Clock.
        # Everything in the ROS environment must synchronize with this clock.
        try:
            keys = og.Controller.Keys
            (self._clock_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": "/ROS_Clock",
                    "evaluator_name": "push",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("readSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("publishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "publishClock.inputs:execIn"),
                        ("readSimTime.outputs:simulationTime", "publishClock.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            logger.error(e)
            self.simulation_app.close()
            exit()

    def one_step(self):
        self.sim_time += 0.01
        self.sim_step += 1
        self.simulation_context.step()

    @staticmethod
    def set_viewports() -> None:
        # vp_3_width, vp_3_height = 640, 480
        # viewportFactory = vp.get_viewport_interface()
        # viewportHandle = viewportFactory.create_instance()
        # viewport_window = viewportFactory.get_viewport_window(viewportHandle)
        # viewport_window.set_window_size(vp_3_width, vp_3_height)
        # viewport_window.set_texture_resolution(vp_3_width, vp_3_height)
        # viewport_window.set_active_camera("/OmniverseKit_Persp")
        # viewport_window.set_camera_position("/OmniverseKit_Persp", 550.0, -150.0, 250.0, True)
        # viewport_window.set_camera_target("/OmniverseKit_Persp", 300.0, 25.0, 50.0, True)
        camera_path = "/World/Camera"
        viewport = get_active_viewport()
        if not viewport:
            raise RuntimeError("No active Viewport")

        # Set the Viewport's active camera to the
        # camera prim path you want to switch to.
        viewport.camera_path = camera_path

    def evaluate_sync_ros2_clock_graph(self, step_size):
        og.Controller.evaluate_sync(self._clock_graph)

    def init_play(self, step_num, tmp_physics_callback_dict=None):
        # omni.timeline.get_timeline_interface().stop()
        # omni.timeline.get_timeline_interface().play()

        if tmp_physics_callback_dict is not None:
            for physics_callback_dict in (self.physics_callback_dict, tmp_physics_callback_dict):
                for physics_callback_name, physics_callback_fn in physics_callback_dict.items():
                    self.simulation_context.add_physics_callback(physics_callback_name, callback_fn=physics_callback_fn)
        else:
            for physics_callback_name, physics_callback_fn in self.physics_callback_dict.items():
                self.simulation_context.add_physics_callback(physics_callback_name, callback_fn=physics_callback_fn)

        # for _ in tqdm.tqdm(range(50)):
        #     self.simulation_context.step()
        # logger.success(f'The scene initialization step is completed.')

        for _ in tqdm.tqdm(range(step_num)):
            self.one_step()
        # logger.success(f'The scene initialization step is completed.')

    def update_sim(self, num_steps):
        for _ in range(num_steps):
            simulation_app.update()
        print("update_out")

    def reset(self):
        self.reset_signal = True

    def _check_abnormal(self):
        if (self._robot.body_prim.is_valid()
                and (abs(self._robot.body_prim.get_world_pose()[0][0]) >= 1e3
                     or abs(self._robot.arm_base_prim.get_world_pose()[0][0] >= 1e3))):
            logger.info("body_prim pose is {}".format((abs(self._robot.body_prim.get_world_pose()[0][0]))))
            logger.info("arm_base_prim pose is {}".format((abs(self._robot.arm_base_prim.get_world_pose()[0][0]))))
            logger.error("NAN, the robot is out of the world")
            return True
        else:
            return False

    def init_robot_posture(self, step_num: int) -> None:
        for _ in range(step_num):
            self._robot.init_robot_posture(step_num, self.robot_init_arm_joints, self.robot_init_gripper_degrees,
                                           self.robot_init_body_position)
            self.one_step()

    def play(self) -> None:
        """Start the simulations."""
        try:
            # add contact tracker to source and target objects
            for task_key in self.contact_report_dict.keys():
                if task_key in self.scene_task_name:
                    self.contact_report_dict[task_key]()

            # timeline starts
            self.update_sim(100)
            omni.timeline.get_timeline_interface().stop()
            omni.timeline.get_timeline_interface().play()

            self.init_play(step_num=200)
            logger.success(f'The scene initialization is completed at sim step {self.sim_step}.')
            self.init_robot_posture(step_num=100)

            # Start sim
            global _LOADED_TASK_EVENT
            _LOADED_TASK_EVENT.set()

            while True:
                # check abnormal
                if self._check_abnormal():
                    break

                # time to quit
                if self.reset_signal:
                    break

                self.one_step()

            omni.timeline.get_timeline_interface().stop()
        except (Exception,):
            logger.error('Error in task play: %s' % traceback.format_exc())

    def shut_down(self, kill_instantly: bool = True):
        """Defines how you prefer when all simulation steps are executed.

        Args:
            kill_instantly: If True, the simulation app will be closed when simulation is finished; if False, the
                simulation app will not close, you can further exam the simulation setups.
        """
        if kill_instantly:
            if hasattr(self, "simulation_context"):
                del self.simulation_context
            if hasattr(self, "_contact_report_sub"):
                del self._contact_report_sub
            simulation_app.close()
        else:
            while self.simulation_app.is_running():
                self.simulation_app.update()


_ROBOT_TASK: Optional[RobotTask] = None
_TASK_RUNNER: Optional[IsaacSimTaskRunner] = None


class TaskHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith('/info'):
            self.do_HEAD()
        else:
            self.send_error(404, 'Path not found')
            return None

    def do_HEAD(self, response_code: int = 200):
        self.send_response(response_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        try:
            global _ROBOT_TASK, _LOAD_TASK_EVENT, _LOADED_TASK_EVENT, _TASK_RUNNER

            content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
            post_data = self.rfile.read(content_length)  # <--- Gets the data itself

            json_message = json.loads(post_data.decode('utf-8'))
            command_code = json_message['command']
            return_code = 200
            result_data = {}
            result_code = '0'
            result_msg = 'ok'
            if command_code == "load_task_simulation":
                _ROBOT_TASK = RobotTask(
                    scene_task_id=json_message['scene_task_id'],
                    scene_usd=json_message['scene_usd'],
                    scene_task_name=json_message['scene_task_name'],
                    source_object_prim_path=json_message['source_object_prim_path'],
                    target_object_prim_path=json_message['target_object_prim_path'],
                    robot_init_pose=np.asarray(json_message['robot_init_pose']).reshape(-1, 3).tolist(),
                    command=json_message['task_command'],
                    robot_init_arm_joints=json_message.get('robot_init_arm_joints', None),
                    robot_init_gripper_degrees=json_message.get('robot_init_gripper_degrees', None),
                    robot_init_body_position=json_message.get('robot_init_body_position', None),
                    init_object_info=json_message.get('init_object_info', None)
                )

                # start load isaac sim
                _LOAD_TASK_EVENT.set()

                # wait for isaac loading
                _LOADED_TASK_EVENT.wait()

            else:
                return_code = 404
                result_code = '404'
                result_msg = 'Service Not Found'

            return_message = {
                'command': command_code,
                'timestamp': datetime.now().strftime(_TIMESTAMP_STRING_FORMAT),
                'result_code': result_code,
                'result_msg': result_msg,
                'result_data': result_data
            }

            self.do_HEAD(return_code)
            self.wfile.write(json.dumps(return_message).encode('utf-8'))
        except (Exception,):
            logger.error('Error in http post handler: %s' % traceback.format_exc())


class IsaacSimTaskLauncher:
    def _listener_http(self) -> None:
        class ThreadingServer(ThreadingMixIn, http.server.HTTPServer):
            pass

        server = ThreadingServer((config['roborpc']['sim_robots']['isaac_sim']['task_host']
                                  , config['roborpc']['sim_robots']['isaac_sim']['task_port']), TaskHTTPRequestHandler)
        server.serve_forever()

    def run(self):
        try:
            logger.info("Start task http handler.")
            global _ROBOT_TASK, _LOAD_TASK_EVENT, _TASK_RUNNER
            threading.Thread(target=self._listener_http, name='TaskHttpListener', daemon=True).start()

            # Scene loading can only be performed on the main thread.
            # Waiting for the HTTP thread to receive the rendering task and perform the scene loading.
            _LOAD_TASK_EVENT.wait()

            if _ROBOT_TASK is not None:
                _TASK_RUNNER = IsaacSimTaskRunner(
                    simulation_app=simulation_app,
                    robot_task=_ROBOT_TASK,
                )

                _TASK_RUNNER.play()
                _TASK_RUNNER.shut_down(True)
        except (Exception,):
            logger.error('Error in TaskLauncher.run(): %s' % traceback.format_exc())


if __name__ == "__main__":
    IsaacSimTaskLauncher().run()
