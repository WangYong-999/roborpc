import threading
import time
from copy import deepcopy
from queue import Queue

import cv2
import numpy as np

from droid.utils.data_utils.parameters import hand_camera_id
from droid.utils.data_utils.time import time_ms
from droid.utils.camera_utils.camera_readers.camera import CameraDriver
from common.config_loader import config as common_config

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: You have not setup the ZED cameras, and currently cannot use them")


def gather_zed_cameras():
    all_zed_cameras = []
    try:
        cameras = sl.Camera.get_device_list()
    except NameError:
        return []

    for cam in cameras:
        cam = ZedCamera(cam)
        all_zed_cameras.append(cam)

    return all_zed_cameras


resize_func_map = {"cv2": cv2.resize, None: None}

standard_params = dict(
    camera_resolution=sl.RESOLUTION.HD720, depth_stabilization=False, camera_fps=60, camera_image_flip=sl.FLIP_MODE.OFF,
    depth_minimum_distance=0.1, depth_mode=sl.DEPTH_MODE.ULTRA, coordinate_units=sl.UNIT.MILLIMETER
)

advanced_params = dict(
    depth_minimum_distance=0.1, camera_resolution=sl.RESOLUTION.HD2K, depth_stabilization=False, camera_fps=15,
    camera_image_flip=sl.FLIP_MODE.OFF
)


class ZedCamera(CameraDriver):
    def __init__(self, camera):
        # Save Parameters #
        self.data_dict = {}
        self.cache_data_dict = Queue()
        self.resizer_resolution = None
        self.zed_resolution = None
        self.skip_reading = None
        self.concatenate_images = None
        self.image = None
        self.resize_func = None
        self.pointcloud = None
        self.depth = None
        self.traj_resolution = None
        self.traj_concatenate_images = None
        self.traj_image = None
        self.serial_number = str(camera.serial_number)
        self.is_hand_camera = self.serial_number == hand_camera_id
        self.high_res_calibration = False
        self.current_mode = None
        self._current_params = None

        self.camera_resolution = None
        self.first_record = True
        self.camera_fps = common_config["droid"]["collector"]["camera_fps"]
        self.depth_video = None
        self.color_video = None
        self.recordingState = False

        self.collect_control_mode = common_config["droid"]["robot"]["control_mode"] == "collector"
        self.save_data_mode = common_config["droid"]["collector"]["save_data_mode"]

        # Open Camera #
        print("Opening Zed: ", self.serial_number)

        # FOR VIDEO RECORD
        self.use_zed_save_mp4 = common_config["droid"]["collector"]["use_zed_save_mp4"]
        if self.use_zed_save_mp4 and self.collect_control_mode and self.save_data_mode == "H5_MP4":
            threading.Thread(target=self.video_loop, args=(), daemon=True).start()

    def read_camera(self):
        if self.skip_reading:
            return {}, {}

        # Read Camera #
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            return None
        timestamp_dict[self.serial_number + "_read_end"] = time_ms()

        # Benchmark Latency #
        received_time = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        timestamp_dict[self.serial_number + "_frame_received"] = received_time
        timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

        # Return Data #
        if (common_config["droid"]["robot"]["control_mode"] == "collector" and
                common_config["droid"]["collector"]["save_data_mode"] == "H5_SVO"
                and common_config["droid"]["collector"]["save_data_speedup"]):
            return self.data_dict, timestamp_dict

        if self.image:
            if self.concatenate_images:
                self._cam.retrieve_image(self._sbs_img, sl.VIEW.SIDE_BY_SIDE, resolution=self.zed_resolution)
                self.data_dict["image"] = {self.serial_number: self._process_frame(self._sbs_img)}
            else:
                self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT, resolution=self.zed_resolution)
                self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT, resolution=self.zed_resolution)
                if common_config["droid"]["collector"]["use_one_side_camera"]:
                    self.data_dict["image"] = {
                        self.serial_number + "_left": self._process_frame(self._left_img)
                    }
                else:
                    self.data_dict["image"] = {
                        self.serial_number + "_left": self._process_frame(self._left_img),
                        self.serial_number + "_right": self._process_frame(self._right_img),
                    }
        if self.depth:
            self._cam.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH, resolution=self.zed_resolution)
            self._cam.retrieve_measure(self._right_depth, sl.MEASURE.DEPTH_RIGHT, resolution=self.zed_resolution)
            if common_config["droid"]["collector"]["use_one_side_camera"]:
                self.data_dict["depth"] = {
                    self.serial_number + '_left': self._left_depth.get_data().copy()}
            else:
                self.data_dict["depth"] = {
                    self.serial_number + '_left': self._left_depth.get_data().copy(),
                    self.serial_number + '_right': self._right_depth.get_data().copy()}
        # if self.pointcloud:
        # 	self._cam.retrieve_measure(self._left_pointcloud, sl.MEASURE.XYZRGBA, resolution=self.zed_resolution)
        # 	self._cam.retrieve_measure(self._right_pointcloud, sl.MEASURE.XYZRGBA_RIGHT, resolution=self.zed_resolution)
        # 	data_dict['pointcloud'] = {
        # 		self.serial_number + '_left': self._left_pointcloud.get_data().copy(),
        # 		self.serial_number + '_right': self._right_pointcloud.get_data().copy()}

        self.cache_data_dict.put(self.data_dict)

        return self.data_dict, timestamp_dict

    def start_recording(self, filename):
        if self.collect_control_mode:
            if self.save_data_mode == "H5_SVO":
                assert filename.endswith(".svo")
                recording_param = sl.RecordingParameters(filename, sl.SVO_COMPRESSION_MODE.H265)
                err = self._cam.enable_recording(recording_param)
                assert err == sl.ERROR_CODE.SUCCESS
            if self.use_zed_save_mp4 and self.save_data_mode == "H5_MP4":
                if self.first_record:
                    filename = str(filename).replace("SVO", "MP4")
                    color_video_file_name = str(filename).replace(".svo", "_color.mp4")
                    depth_video_file_name = str(filename).replace(".svo", "_depth.mp4")
                    print("[INFO] start recording {}".format(color_video_file_name))
                    print("[INFO] start recording {}".format(depth_video_file_name))

                    self.color_video = cv2.VideoWriter(color_video_file_name, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                                       fps=self.camera_fps,
                                                       frameSize=(self.camera_resolution[0], self.camera_resolution[1]))
                    self.depth_video = cv2.VideoWriter(depth_video_file_name, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                                       fps=self.camera_fps,
                                                       frameSize=(self.camera_resolution[0], self.camera_resolution[1]))
                    self.first_record = False
                    self.recordingState = True
        else:
            print("Zed is not in collect_control_mode")

    def stop_recording(self):
        if self.collect_control_mode:
            if self.save_data_mode == "H5_SVO":
                self._cam.disable_recording()
            if self.use_zed_save_mp4 and self.save_data_mode == "H5_MP4":
                if self.recordingState:
                    while not self.cache_data_dict.empty():
                        print(f"wait cache mp4 data {self.cache_data_dict.qsize()} join....")
                        time.sleep(1)
                    print(f"cache mp4 data finished....")
                    self.recordingState = False
                    self.first_record = True
                    self.color_video.release()
                    self.depth_video.release()
        else:
            print("Zed is not in collect_control_mode")

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

    def set_calibration_mode(self):
        # Set Parameters #
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.zed_resolution = sl.Resolution(0, 0)
        self.resizer_resolution = (0, 0)

        # Set Mode #
        change_settings_1 = self.high_res_calibration and (self._current_params != advanced_params)
        change_settings_2 = (not self.high_res_calibration) and (self._current_params != standard_params)
        if change_settings_1:
            self._configure_camera(advanced_params)
        if change_settings_2:
            self._configure_camera(standard_params)
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        # Set Parameters #
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        if self.resize_func is None:
            self.zed_resolution = sl.Resolution(*self.traj_resolution)
            self.resizer_resolution = (0, 0)
        else:
            self.zed_resolution = sl.Resolution(0, 0)
            self.resizer_resolution = self.traj_resolution

        # Set Mode #
        change_settings = self._current_params != standard_params
        if change_settings:
            self._configure_camera(standard_params)
        self.current_mode = "trajectory"

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    def video_loop(self):
        try:
            while True:
                if not self.cache_data_dict.empty() and self.color_video is not None:
                    data_dict = self.cache_data_dict.get()
                    rgb_image = cv2.cvtColor(data_dict["image"][self.serial_number + "_left"],
                                             cv2.COLOR_BGRA2BGR)
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(data_dict["depth"][self.serial_number + "_left"],
                                            alpha=0.03), cv2.COLORMAP_JET)
                    self.color_video.write(rgb_image)
                    self.depth_video.write(depth_colormap)
                else:
                    time.sleep(0.01)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def disable_camera(self):
        if self.current_mode == "disabled":
            return
        if hasattr(self, "_cam"):
            self._current_params = None
            self._cam.close()
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"

    def _configure_camera(self, init_params):
        # Close Existing Camera #
        self.disable_camera()

        # Initialize Readers #
        self._cam = sl.Camera()
        self._sbs_img = sl.Mat()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._left_depth = sl.Mat()
        self._right_depth = sl.Mat()
        self._left_pointcloud = sl.Mat()
        self._right_pointcloud = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        # Open Camera #
        self._current_params = init_params
        sl_params = sl.InitParameters(**init_params)
        sl_params.set_from_serial_number(int(self.serial_number))
        sl_params.camera_image_flip = sl.FLIP_MODE.OFF
        status = self._cam.open(sl_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Camera Failed To Open")

        # Save Intrinsics #
        self.latency = int(2.5 * (1e3 / sl_params.camera_fps))
        if common_config["droid"]["robot"]["control_mode"] == "evaluation":
            width = common_config["droid"]["evaluation"]["image_width"]
            height = common_config["droid"]["evaluation"]["image_height"]
        else:
            width = common_config["droid"]["collector"]["image_width"]
            height = common_config["droid"]["collector"]["image_height"]
        self.camera_resolution = (width, height)
        calib_params = self._cam.get_camera_information(
            sl.Resolution(width, height)).camera_configuration.calibration_parameters
        self._intrinsics = {
            self.serial_number + "_left": self._process_intrinsics(calib_params.left_cam),
            self.serial_number + "_right": self._process_intrinsics(calib_params.right_cam),
        }

    @staticmethod
    def _process_intrinsics(params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])
        intrinsics["distCoeffs"] = np.array(list(params.disto))
        return intrinsics

    def _process_frame(self, frame):
        frame = deepcopy(frame.get_data())
        if self.resizer_resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resizer_resolution)

    def get_point_cloud_position(self, pixel_x, pixel_y, camera_type='left'):
        width = round(self._left_img.get_width() / 2)
        height = round(self._left_img.get_height() / 2)
        if pixel_x > width and pixel_y > height:
            print("pixel out of bound")
            return None
        if camera_type == 'right':
            err, point_cloud_value = self._right_pointcloud.get_value(pixel_x, pixel_y)
        else:
            err, point_cloud_value = self._left_pointcloud.get_value(pixel_x, pixel_y)
        if err == sl.ERROR_CODE.SUCCESS:
            return point_cloud_value
        else:
            return None
