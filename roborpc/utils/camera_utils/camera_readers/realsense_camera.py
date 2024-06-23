import datetime
import os
import threading
import time
from copy import deepcopy
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from droid.utils.data_utils.time import time_ms

from droid.utils.camera_utils.camera_readers.camera import CameraDriver
from common.config_loader import config as common_config


def gather_realsense_cameras():
    all_realsense_cameras = []
    try:
        cameras = get_device_ids()
    except NameError:
        return []

    for cam in cameras:
        cam = RealSenseCamera(cam)
        all_realsense_cameras.append(cam)

    return all_realsense_cameras


def get_device_ids() -> List[str]:

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


resize_func_map = {"cv2": cv2.resize, None: None}


class RealSenseCamera(CameraDriver):

    def __init__(self, device_id):

        self.align = None
        self.camera_resolution = None
        self.use_align_color_depth = None
        self.first_record = False
        self.camera_fps = common_config["droid"]["collector"]["camera_fps"]
        self.depth_video = None
        self.color_video = None
        self.recordingState = None
        self._cam = None
        self.serial_number = str(device_id)
        self.high_res_calibration = False
        self.current_mode = None
        self._current_params = None

        self.initialize()
        # Open Camera #
        print("Opening RealSense: ", self.serial_number)

        # FOR VIDEO RECORD
        if common_config["droid"]["collector"]["save_data_mode"] == "H5_MP4":
            threading.Thread(target=self.video_loop, args=(), daemon=True).start()

    def initialize(self):
        if self.serial_number is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._cam = rs.pipeline()
            config = rs.config()
        else:
            self._cam = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)

        if common_config["droid"]["robot"]["control_mode"] == "evaluation":
            width = common_config["droid"]["evaluation"]["image_width"]
            height = common_config["droid"]["evaluation"]["image_height"]
            self.use_align_color_depth = common_config["droid"]["evaluation"]["use_align_color_depth"]
        else:
            width = common_config["droid"]["collector"]["image_width"]
            height = common_config["droid"]["collector"]["image_height"]
            self.use_align_color_depth = common_config["droid"]["collector"]["use_align_color_depth"]

        self.camera_resolution = (width, height)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 60)
        self._cam.start(config)

        self.align = rs.align(rs.stream.color)

    def read_camera(self):
        """Read a frame from the camera.
        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """

        # Read Camera #
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}

        # Benchmark Latency #
        # received_time = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        # timestamp_dict[self.serial_number + "_frame_received"] = received_time
        # timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

        data_dict = {}

        if self.use_align_color_depth:
            frames = self._cam.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]
            # Render images
            # self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        else:

            frames = self._cam.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]
            depth_image = np.asanyarray(depth_frame.get_data())
            # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if self.image:
            if self.concatenate_images:
                print("Not support concatenate images")
            else:
                data_dict["image"] = {
                    self.serial_number + "_left": color_image
                }
        if self.depth:
            data_dict["depth"] = {
                self.serial_number + '_left': depth_image}
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}

        return data_dict, timestamp_dict

    def start_recording(self, filename):
        if common_config["droid"]["collector"]["save_data_mode"] == "H5_MP4":
            if self.first_record:
                filename = str(filename).replace("SVO", "MP4")
                color_video_file_name = str(filename).replace(".mp4", "_color.mp4")
                depth_video_file_name = str(filename).replace(".mp4", "_depth.mp4")

                self.color_video = cv2.VideoWriter(color_video_file_name, cv2.VideoWriter_fourcc(*'mp4v'),
                                                   self.camera_fps, (self.camera_resolution[0], self.camera_resolution[1]),
                                                   1)
                self.depth_video = cv2.VideoWriter(depth_video_file_name, cv2.VideoWriter_fourcc(*'mp4v'),
                                                   self.camera_fps, (self.camera_resolution[0], self.camera_resolution[1]),
                                                   1)
                self.first_record = False

                print("[INFO] start recording {}".format(color_video_file_name))
                print("[INFO] start recording {}".format(depth_video_file_name))
            self.recordingState = True
        else:
            print("Realsense save_data_mode is not H5_MP4")

    def stop_recording(self):
        if common_config["droid"]["collector"]["save_data_mode"] == "H5_MP4":
            if self.recordingState:
                self.recordingState = False
                self.first_record = True
                self._cam.stop()
                self.initialize()
        else:
            print("Realsense save_data_mode is not H5_MP4")

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
            self,
            image=True,
            depth=True,
            pointcloud=True,
            concatenate_images=False,
            resolution=(0, 0),
            resize_func=None,
    ):
        # Non-Permenant Values #
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution

        # Permenant Values #
        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    ### Camera Modes ###
    def set_calibration_mode(self):
        # Set Parameters #
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False

        # Set Mode #
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        # Set Parameters #
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        self.camera_resolution = self.traj_resolution
        # Set Mode #
        self.current_mode = "trajectory"

    def get_intrinsics(self):
        # Retreive the stream and intrinsic properties for both cameras
        profiles = self._cam.get_active_profile()
        streams = {"left": profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                   "right": profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
        intrinsics = {self.serial_number + "_left": streams["left"].get_intrinsics(),
                      self.serial_number + "_right": streams["right"].get_intrinsics()}

        # Print information about both cameras
        print("Left camera:", intrinsics["left"])
        print("Right camera:", intrinsics["right"])
        return intrinsics

    def video_loop(self):
        try:
            while True:
                if self.recordingState:
                    data_dict, _ = self.read_camera()
                    self.color_video.write(data_dict["image"][self.serial_number + "_left"])
                    self.depth_video.write(data_dict["depth"][self.serial_number + "_left"])
                else:
                    time.sleep(0.01)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def disable_camera(self):
        if self.current_mode == "disabled":
            return
        if hasattr(self, "_cam"):
            self._current_params = None
            self._cam.stop()
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
