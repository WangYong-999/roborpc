import base64
import os
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from queue import Empty, Queue

import cv2
import h5py
import numpy as np

from droid.utils.data_utils.subprocess_utils import run_threaded_command
from common.config_loader import config


def write_full_dict_to_hdf5(hdf5_file, data_dict):
    try:
        for key in data_dict.keys():

            # Examine Data #
            curr_data = data_dict[key]
            if type(curr_data) == list:
                curr_data = np.array(curr_data)
            dtype = type(curr_data)

            # Unwrap If Dictionary #
            if dtype == dict:
                if key not in hdf5_file:
                    hdf5_file.create_group(key)
                write_full_dict_to_hdf5(hdf5_file[key], curr_data)
                continue

            # if key in ["21729895_left", "21729895_right", "29392465_left",
            #            "29392465_right", "10805454_left", "10805454_right"]:
            #     encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), 10]
            #     fmt = ".webp"
            #     _, encode_data = cv2.imencode(fmt, curr_data, encode_param)
            #     curr_data = base64.b64encode(encode_data).decode("utf-8")

            # Make Room For Data #
            if key not in hdf5_file:
                if dtype != np.ndarray:
                    dshape = ()
                else:
                    dtype, dshape = curr_data.dtype, curr_data.shape
                hdf5_file.create_dataset(key, (1, *dshape), maxshape=(None, *dshape), dtype=dtype)
            else:
                hdf5_file[key].resize(hdf5_file[key].shape[0] + 1, axis=0)

            # Save Data #
            hdf5_file[key][-1] = curr_data
    except Exception:
        print('error')


def write_dict_to_hdf5(hdf5_file, data_dict, keys_to_ignore=["image", "depth", "pointcloud"]):
    try:
        for key in data_dict.keys():
            # Pass Over Specified Keys #
            if key in keys_to_ignore:
                continue

            # Examine Data #
            curr_data = data_dict[key]
            if type(curr_data) == list:
                curr_data = np.array(curr_data)
            dtype = type(curr_data)

            # Unwrap If Dictionary #
            if dtype == dict:
                if key not in hdf5_file:
                    hdf5_file.create_group(key)
                write_dict_to_hdf5(hdf5_file[key], curr_data)
                continue

            # Make Room For Data #
            if key not in hdf5_file:
                if dtype != np.ndarray:
                    dshape = ()
                else:
                    dtype, dshape = curr_data.dtype, curr_data.shape
                hdf5_file.create_dataset(key, (1, *dshape), maxshape=(None, *dshape), dtype=dtype)
            else:
                hdf5_file[key].resize(hdf5_file[key].shape[0] + 1, axis=0)

            # Save Data #
            hdf5_file[key][-1] = curr_data
    except Exception:
        print('error')


class TrajectoryWriter:
    def __init__(self, filepath, metadata=None, exists_ok=False):
        self.image_id_buffer = []
        self.depth_video = None
        self.color_video = None
        assert (not os.path.isfile(filepath)) or exists_ok
        self._filepath = filepath
        self._save_images = not config["droid"]["collector"]["use_zed_save_mp4"]
        self._hdf5_file = h5py.File(filepath, "w")
        self._queue_dict = defaultdict(Queue)
        self._video_writers = {}
        self._video_files = {}
        self._open = True
        self.camera_fps = config["droid"]["collector"]["camera_fps"]
        if config["droid"]["robot"]["control_mode"] == "evaluation":
            width = config["droid"]["evaluation"]["image_width"]
            height = config["droid"]["evaluation"]["image_height"]
        else:
            width = config["droid"]["collector"]["image_width"]
            height = config["droid"]["collector"]["image_height"]
        self.camera_resolution = (width, height)

        # Add Metadata #
        if metadata is not None:
            self._update_metadata(metadata)

        # Start HDF5 Writer Thread #
        def hdf5_writer(data):
            if config["droid"]["collector"]["save_data_mode"] == "H5_FULL":
                return write_full_dict_to_hdf5(self._hdf5_file, data)
            else:
                return write_dict_to_hdf5(self._hdf5_file, data)

        run_threaded_command(self._write_from_queue, args=(hdf5_writer, self._queue_dict["hdf5"]))

    def write_timestep(self, timestep):
        if self._save_images:
            self._update_video_files(timestep)
        self._queue_dict["hdf5"].put(timestep)

    def _update_metadata(self, metadata):
        for key in metadata:
            self._hdf5_file.attrs[key] = deepcopy(metadata[key])

    def _write_from_queue(self, writer, queue):
        while self._open:
            try:
                data = queue.get(timeout=1)
            except Empty:
                continue
            writer(data)
            queue.task_done()

    def _update_video_files(self, timestep):
        image_dict = timestep["observation"]["image"]
        depth_dict = timestep["observation"]["depth"]
        for image_id in image_dict:
            # Create Writer And Buffer #
            if image_id not in self.image_id_buffer:
                self._create_video_file(image_id)
                run_threaded_command(
                    self._save_image_loop, args=image_id, daemon=False
                )
                run_threaded_command(
                    self._save_depth_loop, args=image_id, daemon=False
                )
                self.image_id_buffer.append(image_id)

            self._queue_dict[f"image_{image_id}"].put(image_dict[image_id])
            self._queue_dict[f"depth_{image_id}"].put(depth_dict[image_id])

    def _create_video_file(self, video_id):
        filename = str(self._filepath).replace('trajectory.h5', 'recordings/MP4/')
        color_video_file_name = filename + str(video_id) + "_color.mp4"
        depth_video_file_name = filename + str(video_id) + "_depth.mp4"
        print("[INFO] start recording {}".format(color_video_file_name))
        print("[INFO] start recording {}".format(depth_video_file_name))

        self.color_video = cv2.VideoWriter(color_video_file_name, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps=self.camera_fps,
                                           frameSize=(self.camera_resolution[0], self.camera_resolution[1]))
        self.depth_video = cv2.VideoWriter(depth_video_file_name, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps=self.camera_fps,
                                           frameSize=(self.camera_resolution[0], self.camera_resolution[1]))

    def _save_image_loop(self, image_id):
        try:
            while not self._queue_dict[f"image_{image_id}"].empty():
                data_dict = self._queue_dict[f"image_{image_id}"].get()
                rgb_image = cv2.cvtColor(data_dict, cv2.COLOR_BGRA2BGR)
                self.color_video.write(rgb_image)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def _save_depth_loop(self, image_id):
        try:
            while not self._queue_dict[f"depth_{image_id}"].empty():
                data_dict = self._queue_dict[f"depth_{image_id}"].get()
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(data_dict, alpha=0.03), cv2.COLORMAP_JET)
                self.depth_video.write(depth_colormap)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def close(self, metadata=None):
        # Add Metadata #
        if metadata is not None:
            self._update_metadata(metadata)
        # Finish Remaining Jobs #
        # [queue.join() for queue in self._queue_dict.values()]
        for queue in self._queue_dict.values():
            while not queue.empty():
                print(f"wait cache data {queue.qsize()} join....")
                time.sleep(1)
        if self._save_images:
            self.color_video.release()
            self.depth_video.release()

        # Close Video Writers #
        for video_id in self._video_writers:
            self._video_writers[video_id].close()

        # Save Serialized Videos #
        for video_id in self._video_files:
            # Create Folder #
            if "videos" not in self._hdf5_file["observations"]:
                self._hdf5_file["observations"].create_group("videos")

            # Get Serialized Video #
            self._video_files[video_id].seek(0)
            serialized_video = np.asarray(self._video_files[video_id].read())

            # Save Data #
            self._hdf5_file["observations"]["videos"].create_dataset(video_id, data=serialized_video)
            self._video_files[video_id].close()

        # Close File #
        self._hdf5_file.close()
        self._open = False

        print(f"Save Successfully!")
