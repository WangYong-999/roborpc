import threading
import traceback

import carb
import numpy as np
import omni
import omni.graph.core as og
import omni.kit.commands
import omni.kit.primitive.mesh
import omni.kit.viewport_legacy as vp  # Isaac Sim 2022.1.1
import tqdm
from omni.isaac.core import SimulationContext, World
from omni.isaac.core.utils.stage import (add_reference_to_stage,
                                         is_stage_loading)
from omni.isaac.urdf import _urdf

from common.config_loader import config_loader
from common.logger_loader import logger
# from robots.gen2.robot_gen2_vision import RobotGen2Vision as Robot
# from robots.dual_franka.robot_dual_franka import RobotDualFranka as Robot
# from robots.dual_franka.robot_single_franka_vision import RobotSingleFrankaVision as Robot
# from robots.dual_franka.single_franka_interface import SingleFrankaInterface as RpcRobot
from robots.ur.robot_dual_ur_vision import RobotDualURVision as Robot
from robots.ur.dual_ur_interface import DualURInterface as RpcRobot


class RunnerBase:
    def __init__(self,
                 simulation_app,
                 physics_dt: float = 1.0 / 100,
                 render_dt: float = 1.0 / 100,
                 stage_units_in_meters: float = 0.01,
                 environment_path=None,
                 robot_init_position=None,
                 robot_init_orientation=None

                 ):
        """Initiate the simulation.

        Args:
            physics_dt: Physics downtime of the scene.
            render_dt: Render downtime of the scene.
            stage_units_in_meters: The state unit.
        """
        logger.debug('hello from runner base init')
        self.simulation_app = simulation_app
        self.skip_rendering = config_loader.config['isaac_config']['skip_rendering']
        #####
        logger.debug(environment_path)
        logger.debug(robot_init_position)
        logger.debug(robot_init_orientation)
        if environment_path is None:
            self.environment_path = config_loader.config['environment']['usd_path']
        else:
            self.environment_path = environment_path

        self.robot_path = config_loader.config['robot']['usd_path']
        if robot_init_position is None:  # dont use == None
            self.robot_init_position = np.asarray(config_loader.config['robot']['config']['init_position'])
        else:
            self.robot_init_position = np.asarray(robot_init_position)
        if robot_init_orientation is None:
            self.robot_init_orientation = np.asarray(config_loader.config['robot']['config']['init_orientation'])
        else:
            self.robot_init_orientation = np.asarray(robot_init_orientation)

        logger.debug("debug")
        # Setting renderer option.
        if self.skip_rendering:
            self._settings = carb.settings.get_settings()
            self._settings.set("/app/renderer/skipMaterialLoading", True)
            # Smaller number reduces material resolution to avoid out-of-memory, max is 15
            self._settings.set("/rtx-transient/resourcemanager/maxMipCount", 0)

        # Load environment.
        logger.debug("debug")
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
        # Load mobile robot.
        logger.debug("debug")
        self._robot = Robot(simulation_app=simulation_app, simulation_context=self.simulation_context,
                            robot_usd_path=self.robot_path,
                            init_position=self.robot_init_position,
                            init_orientation=self.robot_init_orientation)

        robot_rpc = RpcRobot(simulation_app=simulation_app, simulation_context=self.simulation_context,
                             robot_usd_path=self.robot_path,
                             init_position=self.robot_init_position,
                             init_orientation=self.robot_init_orientation)

        def _listener_rpc() -> None:
            try:
                import zerorpc
                s = zerorpc.Server(robot_rpc)
                s.bind("tcp://0.0.0.0:4244")
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
        # Create new viewport (i.e., vp3).
        vp_3_width, vp_3_height = 640, 480
        viewportFactory = vp.get_viewport_interface()
        viewportHandle = viewportFactory.create_instance()
        viewport_window = viewportFactory.get_viewport_window(viewportHandle)
        viewport_window.set_window_size(vp_3_width, vp_3_height)
        viewport_window.set_texture_resolution(vp_3_width, vp_3_height)
        viewport_window.set_active_camera("/OmniverseKit_Persp")
        viewport_window.set_camera_position("/OmniverseKit_Persp", 550.0, -150.0, 250.0, True)
        viewport_window.set_camera_target("/OmniverseKit_Persp", 300.0, 25.0, 50.0, True)

    # @staticmethod
    # def dock_viewports() -> None:
    #     viewport1 = omni.ui.Workspace.get_window("Viewport")
    #     viewport2 = omni.ui.Workspace.get_window("Viewport 2")  # Error: None
    #     viewport3 = omni.ui.Workspace.get_window("Viewport 3")  # Error: None

    def shut_down(self, kill_instantly: bool = True):
        """Defines how you prefer when all simulation steps are executed.

        Args:
            kill_instantly: If True, the simulation app will be closed when simulation is finished; if False, the
                simulation app will not close, you can further exam the simulation setups.
        """
        if kill_instantly:
            del self.simulation_context
            self.simulation_app.close()
        else:
            while self.simulation_app.is_running():
                self.simulation_app.update()

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
