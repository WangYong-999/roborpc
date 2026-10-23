import os
import threading
import time
from collections import defaultdict
from copy import deepcopy
from queue import Queue, Empty
from typing import Dict, Optional, List, Callable, Union

import cv2
import numpy as np
from PIL import Image
import h5py

from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.kinematics_solver.trajectory_interpolation import action_linear_interpolation
from roborpc.robot_env import RobotEnv


def replay_trajectory(env: RobotEnv, hdf5_filepath: str,
                      read_color: bool = True,
                      read_depth: bool = True,
                      max_width: Optional[int] = 1000,
                      max_height: Optional[int] = 500,
                      aspect_ratio: Optional[float] = 1.5,
                      action_interpolation: bool = False
                      ):
    traj_reader = TrajectoryReader(hdf5_filepath, read_color=read_color, read_depth=read_depth)
    horizon = traj_reader.length()

    camera_ids = [item for sublist in config['roborpc']['cameras']['camera_ids'] for item in sublist]
    robot_ids = [item for sublist in config['roborpc']['robots']['robot_ids'] for item in sublist]
    for i in range(horizon):
        timestep = traj_reader.read_timestep()
        time.sleep(1 / env.env_update_rate)
        robot_obs = {}
        camera_obs = {}
        for key in timestep["observation"]:
            if key in camera_ids:
                camera_obs[key] = timestep["observation"][key]
            if key in robot_ids:
                robot_obs[key] = timestep["observation"][key]
        if camera_obs is not None:
            visualize_timestep(
                camera_obs, max_width=max_width, max_height=max_height, aspect_ratio=aspect_ratio, pause_time=15
            )
        action = timestep["action"]
        for robot_id, action_dict in action.items():
            for action_name, action_value in action_dict.items():
                if type(action_value) == np.ndarray:
                    action_dict[action_name] = action_value.tolist()
        if action_interpolation:
            env.step(action_linear_interpolation(robot_obs, timestep["action"]))
        else:
            env.step(timestep["action"])

def visualize_timestep(camera_obs: Dict,
                       max_width: int = 1000,
                       max_height: int = 500,
                       aspect_ratio: float = 1.5,
                       pause_time: int = 15):
    sorted_image_list = []
    for camera_id, obs in camera_obs.items():
        sorted_image_list.append(obs['color'])
    num_images = len(sorted_image_list)
    max_num_rows = int(num_images ** 0.5)
    for num_rows in range(max_num_rows, 0, -1):
        num_cols = num_images // num_rows
        if num_images % num_rows == 0:
            break
    max_img_width, max_img_height = max_width // num_cols, max_height // num_rows
    if max_img_width > aspect_ratio * max_img_height:
        img_width, img_height = max_img_width, int(max_img_width / aspect_ratio)
    else:
        img_width, img_height = int(max_img_height * aspect_ratio), max_img_height

    img_grid = [[] for i in range(num_rows)]

    for i in range(len(sorted_image_list)):
        img = Image.fromarray(sorted_image_list[i])
        resized_img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        resized_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
        img_grid[i % num_rows].append(np.array(resized_img))

    for i in range(num_rows):
        img_grid[i] = np.hstack(img_grid[i])
    img_grid = np.vstack(img_grid)
    cv2.imshow("Image Feed", img_grid)
    cv2.waitKey(pause_time)


def visualize_trajectory(
        hdf5_filepath: str,
        max_width: Optional[int] = 1000,
        max_height: Optional[int] = 500,
        aspect_ratio: Optional[float] = 1.5
):
    traj_reader = TrajectoryReader(hdf5_filepath, read_color=True, read_depth=True)

    horizon = traj_reader.length()

    camera_ids = [item for sublist in config['roborpc']['cameras']['camera_ids'] for item in sublist]
    for i in range(horizon):
        timestep = traj_reader.read_timestep()
        # if not timestep["observation"]["controller_info"].get("movement_enabled", True):
        #     continue
        camera_obs = {}
        for key in timestep["observation"]:
            if key in camera_ids:
                camera_obs[key] = timestep["observation"][key]
        if camera_obs is not None:
            visualize_timestep(
                camera_obs, max_width=max_width, max_height=max_height, aspect_ratio=aspect_ratio, pause_time=15
            )
    traj_reader.close()


class TrajectoryReader:
    def __init__(self, hdf5_filepath: str, read_color: bool = True, read_depth: bool = True):
        self._hdf5_file = h5py.File(hdf5_filepath, "r")
        self._keys_to_ignore = []
        if not read_color:
            self._keys_to_ignore.append("color")
        if not read_depth:
            self._keys_to_ignore.append("depth")
        self._length = self.get_hdf5_length(self._hdf5_file)
        self._index = 0

    def length(self):
        return self._length

    def get_hdf5_length(self, hdf5_file: Union[h5py.File, h5py.Group]):
        length = None
        for key in hdf5_file.keys():
            if key in self._keys_to_ignore:
                continue

            curr_data = hdf5_file[key]
            if isinstance(curr_data, h5py.Group):
                curr_length = self.get_hdf5_length(curr_data)
            elif isinstance(curr_data, h5py.Dataset):
                curr_length = len(curr_data)
            else:
                raise ValueError

            if length is None:
                length = curr_length
            if not key == "language_distilbert":
                assert curr_length == length, f"key: {key}, curr_length: {curr_length}, length: {length}"
        return length

    def read_timestep(self, index: Optional[int] = None):
        if index is None:
            index = self._index
        else:
            self._index = index
        assert index < self._length
        timestep = self.load_hdf5_to_dict(self._hdf5_file, self._index)

        self._index += 1
        return timestep

    def load_hdf5_to_dict(self, hdf5_file: Union[h5py.File, h5py.Group], index: Optional[int]):
        data_dict = {}
        for key in hdf5_file.keys():
            if key in self._keys_to_ignore:
                continue

            curr_data = hdf5_file[key]
            if isinstance(curr_data, h5py.Group):
                data_dict[key] = self.load_hdf5_to_dict(curr_data, index)
            elif isinstance(curr_data, h5py.Dataset):
                if not key == "language_distilbert":
                    data_dict[key] = curr_data[index]
            else:
                raise ValueError

        return data_dict

    def close(self):
        self._hdf5_file.close()


class TrajectoryWriter:
    def __init__(self, hdf5_filepath: str, metadata: Optional[Dict] = None,
                 save_color: bool = True, save_depth: bool = True):
        self._filepath = hdf5_filepath
        self._hdf5_file = h5py.File(hdf5_filepath, "w")
        self._queue_dict = defaultdict(Queue)
        self._open = True
        if metadata is not None:
            self._update_metadata(metadata)

        self._keys_to_ignore = []
        if not save_color:
            self._keys_to_ignore.append("color")
        if not save_depth:
            self._keys_to_ignore.append("depth")

        def hdf5_writer(data):
            self.write_dict_to_hdf5(self._hdf5_file, data)

        threading.Thread(target=self._write_from_queue,
                         args=(hdf5_writer, self._queue_dict["hdf5"]), daemon=True).start()

    def write_dict_to_hdf5(self, hdf5_file: h5py.File, data_dict: defaultdict):
        try:
            for key in data_dict.keys():
                if key in self._keys_to_ignore:
                    continue

                curr_data = data_dict[key]
                if type(curr_data) == list:
                    curr_data = np.array(curr_data)
                dtype = type(curr_data)

                if dtype == dict:
                    if key not in hdf5_file:
                        hdf5_file.create_group(key)
                    self.write_dict_to_hdf5(hdf5_file[key], curr_data)
                    continue

                if key not in hdf5_file:
                    if dtype != np.ndarray:
                        shape = ()
                    else:
                        dtype, shape = curr_data.dtype, curr_data.shape
                    hdf5_file.create_dataset(key, (1, *shape), maxshape=(None, *shape), dtype=dtype)
                else:
                    hdf5_file[key].resize(hdf5_file[key].shape[0] + 1, axis=0)

                hdf5_file[key][-1] = curr_data
        except Exception as e:
            logger.error(f"Error writing data to hdf5 file: {e}")

    def write_timestep(self, timestep: Dict):
        self._queue_dict["hdf5"].put(timestep)

    def _update_metadata(self, metadata: Dict):
        for key in metadata:
            self._hdf5_file.attrs[key] = deepcopy(metadata[key])

    def _write_from_queue(self, writer: Callable, queue: Queue):
        while self._open:
            try:
                data = queue.get(timeout=1)
            except Empty:
                continue
            writer(data)
            queue.task_done()

    def close(self, metadata=None):
        if metadata is not None:
            self._update_metadata(metadata)

        for queue in self._queue_dict.values():
            while not queue.empty():
                logger.info(f"Waiting cache data {queue.qsize()} join....")
                time.sleep(1)
        time.sleep(2)
        self._open = False
        self._hdf5_file.close()


def crawler(dir_name, filter_func=None):
    subfolders = [f.path for f in os.scandir(dir_name) if f.is_dir()]
    traj_files = [f.path for f in os.scandir(dir_name) if (f.is_file() and "trajectory.h5" in f.path)]

    if len(traj_files):
        # Only Save Desired Data #
        if filter_func is None:
            use_data = True
        else:
            hdf5_file = h5py.File(traj_files[0], "r")
            use_data = filter_func(hdf5_file.attrs)
            hdf5_file.close()

        if use_data:
            return [dir_name]

    all_folder_paths = []
    for child_dir_name in subfolders:
        child_paths = crawler(child_dir_name, filter_func=filter_func)
        all_folder_paths.extend(child_paths)

    return all_folder_paths


def load_trajectory(
        filepath=None,
        remove_skipped_steps=False,
        num_samples_per_traj=None,
        num_samples_per_traj_coeff=1.5,
):
    traj_reader = TrajectoryReader(filepath)

    horizon = traj_reader.length()
    timestep_list = []

    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    for i in indices_to_save:
        timestep = traj_reader.read_timestep(index=i)
        timestep_list.append(timestep)

    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    traj_reader.close()

    return timestep_list
