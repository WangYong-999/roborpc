import os
import time

import matplotlib.pyplot as plt

from droid.robots.sim_env import make_sim_env
from droid.utils.data_utils.parameters import robot_type, robot_serial_number, droid_version
from common.config_loader import config
from droid.utils.trajectory_utils.trajectory_writer import TrajectoryWriter


class MujocoDataCollector:
    def __init__(self, env, controller):
        self.obs_pointer = None
        self.traj_running = None
        self.failure_logdir = None
        self.last_traj_name = None
        self.env = env
        self.controller = controller
        collector_config = config["droid"]["collector"]
        self.horizon = None if collector_config["horizon"] == 0 else collector_config["horizon"]

    def collect_data(self, info=None, practice=False, reset_robot=True, save_images=False):
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
        self.env.establish_connection()
        controller_info = self.collect_trajectory(
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

    def collect_trajectory(
            env,
            controller=None,
            horizon=None,
            save_filepath=None,
            metadata=None,
            wait_for_controller=False,
            obs_pointer=None,
            save_images=False,
            recording_folderpath=False,
            randomize_reset=False,
            reset_robot=True,
    ):
        """
        Collects a robot trajectory.
        - If a horizon is given, we will step the environment accordingly
        - Otherwise, we will end the trajectory when the controller tells us to
        - If you need a pointer to the current observation, pass a dictionary in for obs_pointer
        """
        # Check Parameters #
        assert (controller is not None) or (horizon is not None)
        if wait_for_controller:
            assert controller is not None
        if obs_pointer is not None:
            assert isinstance(obs_pointer, dict)
        if save_images:
            assert save_filepath is not None

        # Reset States #
        if controller is not None:
            controller.reset_state()

        # Prepare Data Writers If Necesary #
        if save_filepath:
            traj_writer = TrajectoryWriter(save_filepath, metadata=metadata)

        # Prepare For Trajectory #
        num_steps = 0

        # Start recording audio
        collect_config = config["droid"]["collector"]
        if config["droid"]["collector"]["save_audio"]:
            # print(sd.query_devices())
            # import ipdb; ipdb.set_trace()
            sd.default.device = "USB PnP Sound Device"
            video_duration = config["droid"]["collector"]["max_timesteps"] * dt  # Total duration of the episode.
            audio_duration = video_duration + 140  # add time to account for latency
            audio_sampling_rate = 48000  # Standard sampling rate for this device
            audio_recording = sd.rec(
                int(audio_duration * audio_sampling_rate),
                samplerate=audio_sampling_rate,
                channels=1,
            )
            audio_start = time.time()

        task_name = collect_config["droid"]["collector"]["sim_task_name"]
        # Begin! #
        print("press button to move arm!")
        while True:
            # setup the environment
            sim_env = make_sim_env(task_name)
            ts = sim_env.reset()

            episode_replay = [ts]
            # setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(joint_traj)):  # note: this will increase episode length by 1
                action = joint_traj[t]
                ts = sim_env.step(action)
                episode_replay.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.02)

            # Collect Miscellaneous Info #
            controller_info = {} if (controller is None) else controller.get_info()
            skip_action = wait_for_controller and (not controller_info["movement_enabled"])
            control_timestamps = {"step_start": time_ms()}

            # Get Observation #
            obs = env.get_observation()
            if obs_pointer is not None:
                obs_pointer.update(obs)
            obs["controller_info"] = controller_info
            obs["timestamp"]["skip_action"] = skip_action

            # Get Action #
            control_timestamps["policy_start"] = time_ms()
            action, controller_action_info = controller.forward(obs, include_info=True)

            # Regularize Control Frequency #
            control_timestamps["sleep_start"] = time_ms()
            comp_time = time_ms() - control_timestamps["step_start"]
            sleep_left = (1 / env.control_hz) - (comp_time / 1000)
            if sleep_left > 0:
                time.sleep(sleep_left)

            # Step Environment #
            control_timestamps["control_start"] = time_ms()
            if skip_action:
                print(f"skip action: {action}")
                action_info = env.create_action_dict(np.zeros_like(action), action_space=env.action_space)
            else:
                print(f"action: {action}")
                action_info = env.step(action)
            action_info.update(controller_action_info)

            # Save Data #
            control_timestamps["step_end"] = time_ms()
            obs["timestamp"]["control"] = control_timestamps
            timestep = {"observation": obs, "action": action_info}
            if save_filepath:
                traj_writer.write_timestep(timestep)

            # Check Termination #
            num_steps += 1
            if horizon is not None:
                end_traj = horizon == num_steps
            else:
                end_traj = controller_info["success"] or controller_info["failure"]

            # Close Files And Return #
            if end_traj:
                plt.close()
                if recording_folderpath:
                    env.camera_reader.stop_recording()
                if save_filepath:
                    traj_writer.close(metadata=controller_info)
                return controller_info