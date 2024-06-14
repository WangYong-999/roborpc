import http
import http.server
import json
import numpy as np

np.set_printoptions(precision=4)
import threading
import traceback

from datetime import datetime
from functools import partial
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Optional

from common.config_loader import config_loader

from omni.isaac.kit import SimulationApp

CONFIG = {"width": 1280, "height": 720, "sync_loads": False,
          "headless": config_loader.config['isaac_config']['headless'], "renderer": "RayTracedLighting"}
simulation_app = SimulationApp(CONFIG)  # we can also run as headless.

import omni
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import is_stage_loading
import omni.kit.commands
import omni.kit.primitive.mesh
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.physx.scripts import physicsUtils
from omni.physx.scripts.physicsUtils import *

enable_extension("omni.isaac.ros2_bridge")
# enable_extension("midea.robot.sim-0.1.0")
# enable_extension("omni.services.streamclient.websocket")  # http://<server IP address>:8211/streaming/client

from common.logger_loader import logger

try:
    from tasks.data_collector.task_data_collector import TaskDataCollector
    from tasks.helper.robot_task import RobotTask
    from tasks.runner_base import RunnerBase
    from tasks.retriever.retriever import Retriever
    from helper.usd.utils import Get_Type_Prims
except (Exception,):
    logger.error('Error in import: %s' % traceback.format_exc())

try:
    while is_stage_loading():
        simulation_app.update()
except (Exception,):
    logger.error('Error in init: %s' % traceback.format_exc())

# jinzhao thres
label2id_dict = {
    "Plate": 1,
    "Cup": 2,
    "Bottle_Drink": 3,
    "Can": 4,
    "Sauce": 5,
    "Fruit": 6
}

_TIMESTAMP_STRING_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
_LOAD_TASK_EVENT: threading.Event = threading.Event()
_LOADED_TASK_EVENT: threading.Event = threading.Event()


class RobotRunner(RunnerBase):
    def __init__(self, robot_task: RobotTask) -> None:
        """Initiate the simulation."""
        try:
            logger.debug(config_loader.config)
            robot_init_pose = np.array(robot_task.robot_init_pose)
            super(RobotRunner, self).__init__(simulation_app=simulation_app,
                                              environment_path=robot_task.scene_usd,
                                              robot_init_position=robot_init_pose[0],
                                              robot_init_orientation=robot_init_pose[1]
                                              )
            # self.robot_data_collector = TaskDataCollector(self._robot)

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

            self.retriever = Retriever(self.scene_dir, self.source_object_prim_path,
                                       self.target_object_prim_path, self._robot)
            if self.retriever.is_loaded_scene_json:
                # lcm
                physics_material = PhysicsMaterial(
                    prim_path="/PhysicsMaterial",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.,
                )
                collision_geom = Get_Type_Prims(
                    root=get_prim_at_path("/grip_scene/grabable_items")).get_desired_type_prims()
                for value in collision_geom:
                    if "Plate" in value:
                        collision_geom.pop(collision_geom.index(value))
                for value in collision_geom:
                    physicsUtils.add_physics_material_to_prim(self._stage, get_prim_at_path(value), "/PhysicsMaterial")
                    omni.kit.commands.execute("ChangeProperty",
                                              prop_path="/PhysicsMaterial.physxMaterial:frictionCombineMode",
                                              value="Max",
                                              prev=None)

                self.contact_report_dict = {
                    "near": partial(self.retriever.add_near_obj_contact_report),
                    "receptacle": partial(self.retriever.add_on_receptacle_report)
                }
            else:
                self.contact_report_dict = {}

            while is_stage_loading():
                simulation_app.update()

            logger.success("task runner initialization is Done!")
        except (Exception,):
            logger.error('Error in task runner initialization: %s' % traceback.format_exc())

    def update_sim(self, num_steps):
        for _ in range(num_steps):
            simulation_app.update()
        print("update_out")

    def refresh_task_info(self, task_dict) -> bool:
        self.task_id = task_dict["task_id"]
        self.task_name = task_dict["task_name"]
        logger.info("task name is {}".format(self.task_name))

        is_valid, self.source_object_prim_path, self.target_object_prim_path = self.retriever.refresh_task_info(
            task_dict["source_object_name"], task_dict["target_object_name"])

        return is_valid

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

                # self.robot_data_collector.collect_data(sim_step=self.sim_step, label2id_dict=label2id_dict)

                # self._robot.sub_cmds()

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
_TASK_RUNNER: Optional[RobotRunner] = None


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

                # init record folder
                # _TASK_RUNNER.robot_data_collector.init_scene_task_folder(json_message['scene_task_id'],
                #                                                          json_message['scene_task_name'],
                #                                                          json_message['scene_usd'],
                #                                                          json_message['source_object_prim_path'],
                #                                                          json_message['target_object_prim_path'],
                #                                                          json_message['robot_init_pose'],
                #                                                          json_message['task_command'],
                #                                                          json_message['collector_path'])
            elif command_code == "reset":
                _TASK_RUNNER.reset()
            elif command_code == "refresh_task_info":
                result_data = _TASK_RUNNER.refresh_task_info(task_dict=json_message['task_dict'])
            elif command_code == "start_task_collector":
                _TASK_RUNNER.robot_data_collector.start_collector(rate=json_message['collector']['rate'],
                                                                  scene_task_id=str(_TASK_RUNNER.task_id),
                                                                  command=_TASK_RUNNER.task_name,
                                                                  source_object_name=json_message['collector'][
                                                                      'source_object_name'],
                                                                  target_object_name=json_message['collector'][
                                                                      'target_object_name'])
            elif command_code == "stop_task_collector":
                _TASK_RUNNER.robot_data_collector.stop_collector()
                _TASK_RUNNER.robot_data_collector.save_finish_info(task_status=json_message['task_status'])
            elif command_code == "terminate_failed_task_collector":
                _TASK_RUNNER.robot_data_collector.stop_collector()
                _TASK_RUNNER.robot_data_collector.save_fail_info(failure_info=json_message['failure_info'])
            elif command_code == "retrieve_object_placement_info_at_arm_base":
                result_data = _TASK_RUNNER.retriever.retrieve_object_placement_info(type=json_message['type'],
                                                                                    source_obj=json_message[
                                                                                        'source_obj'],
                                                                                    target_obj=json_message[
                                                                                        'target_obj'],
                                                                                    target_xform=_TASK_RUNNER._robot.arm_base_prim)
            elif command_code == "retrieve_object_bbox_info":
                object_segm_info, object_bbox_info = _TASK_RUNNER.retriever.retrieve_object_bbox_info(
                    _TASK_RUNNER._robot, object_name=json_message['object_name'])
                result_data = {'object_segm_info': object_segm_info.tolist(), 'object_bbox_info': object_bbox_info}
            elif command_code == "retrieve_object_grasp_info":
                result_data = _TASK_RUNNER.retriever.retrieve_object_grasp_info(
                    prim_path=json_message['object_prim_path'], target_xform=_TASK_RUNNER._robot.head_cam_xform)
            elif command_code == "check_source_on_target_aabb":
                result_data = {'check_result': _TASK_RUNNER.retriever.check_source_on_target_aabb()}
            elif command_code == "check_source_in_target_aabb":
                result_data = {'check_result': _TASK_RUNNER.retriever.check_source_in_target_aabb()}
            elif command_code == "check_pick_object":
                result_data = {'check_result': _TASK_RUNNER.retriever.check_pick_obj()}
            elif command_code == "check_object_into_roi":
                result_data = {'check_result': _TASK_RUNNER.retriever.check_place_obj_on_roi()}
            elif command_code == "check_object_near_object":
                result_data = {'check_result': _TASK_RUNNER.retriever.check_place_obj_near_obj()}
            elif command_code == "check_object_into_receptacle":
                result_data = {'check_result': _TASK_RUNNER.retriever.check_place_obj_receptacle()}
            elif command_code == "set_scene_objects_pose":
                result_data = {
                    'check_result': _TASK_RUNNER.retriever.set_scene_objects_pose(json_message['object_pose_dict'])}
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


class TaskLauncher:
    def _listener_http(self) -> None:
        class ThreadingServer(ThreadingMixIn, http.server.HTTPServer):
            pass

        server = ThreadingServer((config_loader.config['http_server_config']['host']
                                  , config_loader.config['http_server_config']['port']), TaskHTTPRequestHandler)
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
                _TASK_RUNNER = RobotRunner(robot_task=_ROBOT_TASK)

                _TASK_RUNNER.play()
                _TASK_RUNNER.shut_down(True)
        except (Exception,):
            logger.error('Error in TaskLauncher.run(): %s' % traceback.format_exc())


if __name__ == "__main__":
    TaskLauncher().run()
