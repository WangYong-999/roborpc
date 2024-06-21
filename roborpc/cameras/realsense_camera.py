import time
from typing import Dict, List, Optional

import numpy as np

from roborpc.cameras.camera_base import CameraBase
import pyrealsense2 as rs
import cv2

from roborpc.cameras.camera_utils import rgb_to_base64, depth_to_base64


class RealSenseCamera(CameraBase):
    def __init__(self, device_id: str):
        super().__init__()
        self.device_id = device_id
        self._pipeline = None
        self.cfg = None

    def connect_now(self):
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.device_id)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cfg = self._pipeline.start(config)

    def disconnect_now(self):
        self._pipeline.stop()
        self._pipeline = None
        self.device_id = None

    def get_device_ids(self) -> List[str]:
        pass

    def get_camera_intrinsics(self) -> Dict[str, Dict[str, List[float]]]:
        def _process_intrinsics(params):
            return {"cameraMatrix": [[params.fx, 0, params.ppx], [0, params.fy, params.ppy], [0, 0, 1]],
                    "distCoeffs": list(params.coeffs)}

        profile_color = self.cfg.get_stream(rs.stream.color)
        profile_depth = self.cfg.get_stream(rs.stream.depth)
        intrinsics_color = profile_color.as_video_stream_profile().get_intrinsics()
        intrinsics_depth = profile_depth.as_video_stream_profile().get_intrinsics()

        intrinsics = {
            self.device_id + "_color": _process_intrinsics(intrinsics_color),
            self.device_id + "_depth": _process_intrinsics(intrinsics_depth),
        }

        # calibration_file = "calibration/" + f"{self._device_id}_intrinsics.json"
        # with open(calibration_file, "w") as jsonFile:
        #     json.dump(self.intrinsics, jsonFile)
        return intrinsics

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        pass

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        image = color_image[:, :, ::-1]
        depth = depth_image
        camera_info = {"image": {
            self.device_id + "_color": rgb_to_base64(image),
        }, "depth": {
            self.device_id + "_depth": rgb_to_base64(depth),
        }}
        return camera_info


class MultiRealSenseCamera(CameraBase):
    def __init__(self, device_ids: Optional[List[str]] = None):
        if device_ids is None:
            device_ids = self.get_device_ids()
        self.cameras = [RealSenseCamera(device_id) for device_id in device_ids]
        self.intrinsics = {}
        self.extrinsics = {}

    def connect_now(self):
        for camera in self.cameras:
            camera.connect_now()
            self.intrinsics.update(camera.get_camera_intrinsics())
            self.extrinsics.update(camera.get_camera_extrinsics())

    def disconnect_now(self):
        for camera in self.cameras:
            camera.disconnect_now()

    def get_device_ids(self) -> List[str]:
        ctx = rs.context()
        devices = ctx.query_devices()
        device_ids = []
        for dev in devices:
            dev.hardware_reset()
            device_ids.append(dev.get_info(rs.camera_info.serial_number))
        time.sleep(2)
        return device_ids

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        return self.intrinsics

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        return self.extrinsics

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        camera_info = {}
        for camera in self.cameras:
            camera_info.update(camera.read_camera())
        return camera_info
