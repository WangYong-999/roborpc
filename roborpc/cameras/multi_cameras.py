import asyncio
from typing import Dict, List

from roborpc.cameras.camera_base import CameraBase
from roborpc.cameras.realsense_camera import RealSenseCamera
from roborpc.cameras.zed_camera import ZedCamera
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config


class MultiCameras(CameraBase):
    def __init__(self):
        self.cameras = {}
        self.camera_config = config['roborpc']['cameras']
        self.camera_ids = self.camera_config['camera_ids'][0]
        self.loop = asyncio.get_event_loop()

    def connect_now(self):
        for camera_id in self.camera_ids:
            if str(camera_id).startswith('realsense_camera'):
                camera_resolution = self.camera_config['realsense_camera'][camera_id]['camera_resolution']
                camera_fps = self.camera_config['realsense_camera'][camera_id]['camera_fps']
                self.cameras[camera_id] = RealSenseCamera(camera_id, camera_resolution, camera_fps)
                self.cameras[camera_id].connect_now()
            elif str(camera_id).startswith('zed_camera'):
                camera_resolution = self.camera_config['zed_camera'][camera_id]['camera_resolution']
                camera_fps = self.camera_config['zed_camera'][camera_id]['camera_fps']
                self.cameras[camera_id] = ZedCamera(camera_id, camera_resolution, camera_fps)
                self.cameras[camera_id].connect_now()
            else:
                logger.error(f"Camera with id {camera_id} not found.")
                exit()
            logger.info(f"Camera with id {camera_id} connected successfully.")

    def disconnect_now(self):
        for camera_id, camera in self.cameras.items():
            self.cameras[camera_id].disconnect_now()
            logger.info(f"Camera with id {camera_id} disconnected successfully.")

    def get_device_ids(self) -> List[str]:
        return self.camera_ids

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        camera_intrinsics = {}
        for camera_id, camera in self.cameras.items():
            camera_intrinsics[camera_id] = asyncio.ensure_future(camera.get_camera_intrinsics())
        self.loop.run_until_complete(asyncio.gather(*camera_intrinsics.values()))
        for camera_id, intrinsics in camera_intrinsics.items():
            camera_intrinsics[camera_id] = intrinsics.result()
        return camera_intrinsics

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        camera_extrinsics = {}
        for camera_id, camera in self.cameras.items():
            camera_extrinsics[camera_id] = asyncio.ensure_future(camera.get_camera_extrinsics())
        self.loop.run_until_complete(asyncio.gather(*camera_extrinsics.values()))
        for camera_id, extrinsics in camera_extrinsics.items():
            camera_extrinsics[camera_id] = extrinsics.result()
        return camera_extrinsics

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        camera_frames = {}
        for camera_id, camera in self.cameras.items():
            camera_frames[camera_id] = asyncio.ensure_future(camera.read_camera())
        self.loop.run_until_complete(asyncio.gather(*camera_frames.values()))
        for camera_id, frame in camera_frames.items():
            camera_frames[camera_id] = frame.result()
        return camera_frames
