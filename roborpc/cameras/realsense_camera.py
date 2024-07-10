import time
import numpy as np
import pyrealsense2 as rs
from typing import Dict, List

from roborpc.cameras.camera_base import CameraBase
from roborpc.cameras.transformations import rgb_to_base64, depth_to_base64


ctx = rs.context()
devices = ctx.query_devices()
DEVICE_IDS = []
for dev in devices:
    dev.hardware_reset()
    DEVICE_IDS.append(dev.get_info(rs.camera_info.serial_number))
time.sleep(2)


class RealSenseCamera(CameraBase):
    def __init__(self, device_id: str, camera_resolution: List[int], fps: int = 30):
        super().__init__()
        self.device_id = device_id
        self.camera_resolution = camera_resolution
        self.fps = fps
        self.pipeline = None
        self.cfg = None

    def connect_now(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        all_devices = DEVICE_IDS
        if self.device_id not in all_devices:
            raise ValueError(f"Device {self.device_id} not found. Available devices: {all_devices}")
        print(f"Using device {self.device_id}")
        config.enable_device(self.device_id)

        config.enable_stream(rs.stream.depth, self.camera_resolution[0], self.camera_resolution[1], rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.camera_resolution[0], self.camera_resolution[1], rs.format.bgr8, self.fps)
        self.cfg = self.pipeline.start(config)

    def disconnect_now(self):
        self.pipeline.stop()
        self.pipeline = None
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
            "color": _process_intrinsics(intrinsics_color),
            "depth": _process_intrinsics(intrinsics_depth),
        }

        # calibration_file = "calibration/" + f"{self._device_id}_intrinsics.json"
        # with open(calibration_file, "w") as jsonFile:
        #     json.dump(self.intrinsics, jsonFile)
        return intrinsics

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        pass

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        image = color_image[:, :, ::-1]
        depth = depth_image
        camera_info = {
            "color": rgb_to_base64(image, quality=100),
            "depth": depth_to_base64(depth, quality=100)
        }
        return camera_info
