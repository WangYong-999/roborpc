import os
import time
from copy import deepcopy
from datetime import date

import cv2
import h5py

import droid.utils.trajectory_utils.trajectory_utils as tu
from droid.utils.calibration_utils.calibration_utils import check_calibration_info
from roborpc.common.config_loader import config

from roborpc.robot_env import RobotEnv
from roborpc.controllers.gello_controller import MultiGelloController


class DataCollector:
    def __init__(self, env: RobotEnv, controller: MultiGelloController):
        self.last_traj_name = None
        self.env = env
        self.controller = controller

        collector_config = config["roborpc"]["collector"]["data_collector"]
        data_dir = collector_config["save_data_dir"]
        self.horizon = None if collector_config["horizon"] == 0 else collector_config["horizon"]
        if data_dir == "":
            dir_path = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(dir_path, "../../data")

        self.last_traj_path = None
        self.traj_running = False
        self.traj_saved = False
        self.obs_pointer = {}
        self.robot_type = config["roborpc"]["robots"]["robot_type"]
        self.robot_serial_number = config["roborpc"]["robots"]["robot_serial_number"]

        # Get Camera Info #
        self.cam_ids = env.camera.get_device_ids()
        self.cam_ids.sort()

        self.success_logdir = os.path.join(data_dir, "success", str(date.today()))
        self.failure_logdir = os.path.join(data_dir, "failure", str(date.today()))
        if not os.path.isdir(self.success_logdir):
            os.makedirs(self.success_logdir)
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)

    def collect_trajectory(self, info=None, practice=False, reset_robot=True, save_images=False):
        self.last_traj_name = time.asctime().replace(" ", "_")

        if info is None:
            info = {}
        info["time"] = self.last_traj_name
        info["robot_serial_number"] = "{0}-{1}".format(robot_type, robot_serial_number)
        info["version_number"] = droid_version

        if practice:
            save_filepath = None
            recording_folderpath = None
        else:
            save_filepath = os.path.join(self.failure_logdir, info["time"], "trajectory.h5")
            recording_folderpath = os.path.join(self.failure_logdir, info["time"], "recordings")
            if not os.path.isdir(recording_folderpath):
                os.makedirs(recording_folderpath)

        # Collect Trajectory #
        self.traj_running = True
        if config["droid"]["robot"]["robot_mode"] == 'real':
            self.env.establish_connection()
        controller_info = tu.collect_trajectory(
            self.env,
            controller=self.controller,
            horizon=self.horizon,
            metadata=info,
            obs_pointer=self.obs_pointer,
            reset_robot=reset_robot,
            recording_folderpath=recording_folderpath,
            save_filepath=save_filepath,
            save_images=save_images,
            wait_for_controller=True,
        )
        self.traj_running = False
        self.obs_pointer = {}

        # Sort Trajectory #
        self.traj_saved = controller_info["success"] and (save_filepath is not None)

        if self.traj_saved:
            self.last_traj_path = os.path.join(self.success_logdir, info["time"])
            os.rename(os.path.join(self.failure_logdir, info["time"]), self.last_traj_path)

        return controller_info["success"]

