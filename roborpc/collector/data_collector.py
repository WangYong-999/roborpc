import os
import threading
import time
from copy import deepcopy
from datetime import date
from typing import Optional, Dict

import cv2
from roborpc.collector.data_collector_utils import TrajectoryWriter, visualize_timestep
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.kinematics_solver.trajectory_interpolation import action_linear_interpolation

from roborpc.robot_env import RobotEnv


class DataCollector:
    def __init__(self, env: RobotEnv):
        self.action_interpolation = None
        self.camera_obs = None
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

        self.success_logdir = os.path.join(data_dir, "success", str(date.today()))
        self.failure_logdir = os.path.join(data_dir, "failure", str(date.today()))
        if not os.path.isdir(self.success_logdir):
            os.makedirs(self.success_logdir)
            logger.info("Created directory for successful trajectories: {}".format(self.success_logdir))
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)
            logger.info("Created directory for failed trajectories: {}".format(self.failure_logdir))
        threading.Thread(target=self.visualize_timestep_loop, daemon=False).start()

    def visualize_timestep_loop(self):
        while True:
            if self.camera_obs is None:
                time.sleep(0.001)
                continue
            try:
                visualize_timestep(self.camera_obs)
            except Exception as e:
                logger.error(f"Error in visualize_timestep_loop: {e}")
                cv2.destroyAllWindows()
                break

    def collect_trajectory(self, info=None, practice=False, reset_robot=True,
                           random_reset=False, action_interpolation=False):
        self.last_traj_name = time.asctime().replace(" ", "_")
        self.action_interpolation = action_interpolation

        info_time = self.last_traj_name

        if practice:
            save_filepath = None
            recording_folder_path = None
        else:
            save_filepath = os.path.join(self.failure_logdir, info_time, "trajectory.h5")
            recording_folder_path = os.path.join(self.failure_logdir, info_time)
            if not os.path.isdir(recording_folder_path):
                os.makedirs(recording_folder_path)

        # Collect Trajectory #
        self.traj_running = True
        controller_info = self.collect_one_trajectory(
            hdf5_file=save_filepath,
            horizon=self.horizon,
            metadata=None,
            random_reset=random_reset,
            reset_robot=reset_robot,
        )
        self.traj_running = False
        self.obs_pointer = {}

        # Sort Trajectory #
        controller_success = False
        for controller_id, info in controller_info.items():
            if info.get("success", True):
                controller_success = True

        if self.horizon is not None:
            logger.info("press A button to save")
            while True:
                controller_info = self.env.controllers.get_info()
                controller_success = None
                for controller_id, info in controller_info.items():
                    if info["success"]:
                        print(f"controller {controller_id} success")
                        controller_success = True
                    if info["failure"]:
                        controller_success = False
                if controller_success is not None:
                    break
        self.traj_saved = controller_success and save_filepath is not None

        if self.traj_saved:
            self.last_traj_path = os.path.join(self.success_logdir, info_time)
            os.rename(os.path.join(self.failure_logdir, info_time), self.last_traj_path)
            logger.success("Trajectory saved to: {}".format(self.last_traj_path))
        else:
            logger.success("Trajectory saved to: {}".format(save_filepath))
        return controller_success

    def collect_one_trajectory(
            self,
            horizon: Optional[int] = None,
            hdf5_file: Optional[str] = None,
            metadata: Optional[Dict] = None,
            random_reset: bool = False,
            reset_robot: bool = True,
    ):
        traj_writer = None
        controller = self.env.controllers
        if hdf5_file:
            traj_writer = TrajectoryWriter(hdf5_file, metadata=metadata)

        num_steps = 0
        if reset_robot:
            self.env.reset(random_reset=random_reset)

        # Begin! #
        logger.info("press A button to move arm!")
        while True:
            controller_info = {} if (controller is None) else controller.get_info()
            move = False
            for controller_id, info in controller_info.items():
                if info.get("success", True):
                    move = True
            if move:
                break

        while True:
            controller_info = {} if (controller is None) else controller.get_info()
            control_timestamps = {"step_start": time.time_ns() / 1_000_000}

            robot_obs, camera_obs = self.env.get_observation()
            self.camera_obs = deepcopy(camera_obs)
            timestep = {"observation": camera_obs, "action": {}}

            control_timestamps["policy_start"] = time.time_ns() / 1_000_000
            action = controller.forward(robot_obs)

            control_timestamps["sleep_start"] = time.time_ns() / 1_000_000
            comp_time = time.time_ns() / 1_000_000 - control_timestamps["step_start"]
            sleep_left = (1 / self.env.env_update_rate) - (comp_time / 1000)
            if sleep_left > 0:
                time.sleep(sleep_left)

            control_timestamps["control_start"] = time.time_ns() / 1_000_000
            if self.action_interpolation:
                action_info = self.env.step(action_linear_interpolation(robot_obs, action))
            else:
                action_info = self.env.step(action)

            control_timestamps["step_end"] = time.time_ns() / 1_000_000
            robot_obs["timestamp"] = {"control": control_timestamps}
            # logger.info(f"control timestamps: {control_timestamps}")
            timestep["observation"].update(robot_obs)
            timestep["action"].update(action_info)
            logger.info(f"robot_obs: {robot_obs}")
            logger.info(f"action_info: {action_info}")
            if hdf5_file:
                traj_writer.write_timestep(timestep)

            num_steps += 1
            end_traj = False
            for controller_id, info in controller_info.items():
                if info.get("success", True) and horizon is None:
                    end_traj = True
                if info.get("failure", False):
                    end_traj = True
            if horizon is not None:
                end_traj = horizon == num_steps
            if end_traj:
                logger.info("Trajectory ended.")
                if hdf5_file:
                    traj_writer.close()
                return controller_info
