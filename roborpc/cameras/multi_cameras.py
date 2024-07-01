import asyncio
from typing import Dict, List

from roborpc.cameras.camera_base import CameraBase
from roborpc.cameras.realsense_camera import RealSenseCamera
from roborpc.cameras.zed_camera import ZedCamera
from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config


class MultiCameras(CameraBase):
    def __init__(self, camera_ids=None):
        self.cameras = {}
        self.camera_config = config['roborpc']['cameras']
        if camera_ids is not None:
            self.camera_ids = camera_ids
        else:
            self.camera_ids = self.camera_config['camera_ids'][0]
        # self.loop = asyncio.get_event_loop()

    def connect_now(self):
        for camera_id in self.camera_ids:
            if str(camera_id).startswith('realsense_camera'):
                camera_resolution = self.camera_config['realsense_camera'][camera_id]['camera_resolution']
                camera_fps = self.camera_config['realsense_camera'][camera_id]['camera_fps']
                camera_serial_number = self.camera_config['realsense_camera'][camera_id]['camera_serial_number']
                print(f"Connecting to RealSense camera with serial number {camera_serial_number}.")
                self.cameras[camera_id] = RealSenseCamera(camera_serial_number, camera_resolution, camera_fps)
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
            camera_intrinsics[camera_id] = camera.get_camera_intrinsics()
        return camera_intrinsics

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        camera_extrinsics = {}
        for camera_id, camera in self.cameras.items():
            camera_extrinsics[camera_id] = camera.get_camera_extrinsics()
        return camera_extrinsics

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        camera_frames = {}
        for camera_id, camera in self.cameras.items():
            camera_frames[camera_id] = camera.read_camera()
        return camera_frames


if __name__ == '__main__':
    import zerorpc
    import argparse

    parser = argparse.ArgumentParser(description='Multi-camera server')
    parser.add_argument('--rpc_port', type=int, default=None, help='RPC port number')
    parser.add_argument('--camera_ids', type=str, default=None, help='Camera IDs to connect to')
    rpc_port = parser.parse_args().rpc_port
    camera_ids = parser.parse_args().camera_ids

    multi_camera = MultiCameras(list(camera_ids.split(',')))
    s = zerorpc.Server(multi_camera)
    if rpc_port is None:
        rpc_port = multi_camera.camera_config['server_rpc_ports'][0]
    logger.info(f"RPC server started on port {rpc_port}")
    s.bind(f"tcp://0.0.0.0:{rpc_port}")
    s.run()
