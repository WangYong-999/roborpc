import os
import time
from datetime import date

from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.collector.data_collector_utils import collect_trajectory

from roborpc.robot_env import RobotEnv


class DataCollector:
    def __init__(self, env: RobotEnv):
        self.last_traj_name = None
        self.env = env

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
            logger.info("Created directory for successful trajectories: {}".format(self.success_logdir))
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)
            logger.info("Created directory for failed trajectories: {}".format(self.failure_logdir))

    def collect_trajectory(self, info=None, practice=False, reset_robot=True, random_reset=False):
        self.last_traj_name = time.asctime().replace(" ", "_")

        if info is None:
            info = {}
        info["time"] = self.last_traj_name

        if practice:
            save_filepath = None
            recording_folder_path = None
        else:
            save_filepath = os.path.join(self.failure_logdir, info["time"], "trajectory.h5")
            recording_folder_path = os.path.join(self.failure_logdir, info["time"], "recordings")
            if not os.path.isdir(recording_folder_path):
                os.makedirs(recording_folder_path)

        # Collect Trajectory #
        self.traj_running = True
        controller_info = collect_trajectory(
            self.env,
            hdf5_file=save_filepath,
            horizon=self.horizon,
            metadata=info,
            random_reset=random_reset,
            reset_robot=reset_robot,
        )
        self.traj_running = False
        self.obs_pointer = {}

        # Sort Trajectory #
        self.traj_saved = controller_info["success"] and (save_filepath is not None)

        if self.traj_saved:
            self.last_traj_path = os.path.join(self.success_logdir, info["time"])
            os.rename(os.path.join(self.failure_logdir, info["time"]), self.last_traj_path)

        return controller_info["success"]

