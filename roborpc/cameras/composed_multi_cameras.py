import asyncio
import pickle
import time
from typing import Dict, List

import numpy as np
import zerorpc

from roborpc.cameras.camera_base import CameraBase
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.cameras.multi_cameras import MultiCameras


class MultiCamerasRpc(CameraBase):
    def __init__(self, server_ip_address: str, server_port: int):
        super().__init__()
        self.server_ip_address = server_ip_address
        self.server_port = server_port
        self.cameras = None

    def connect_now(self):
        self.cameras = zerorpc.Client(heartbeat=100)
        self.cameras.connect("tcp://{}:{}".format(self.server_ip_address, self.server_port))
        self.cameras.connect_now()

    def disconnect_now(self):
        self.cameras.disconnect_now()
        self.cameras.close()

    def get_device_ids(self) -> List[str]:
        return self.cameras.get_device_ids()

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        return self.cameras.get_camera_intrinsics()

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        return self.cameras.get_camera_extrinsices()

    def read_camera(self) -> Dict[str, Dict[str, bytes]]:
        return self.cameras.read_camera()


class ComposedMultiCameras(CameraBase):
    def __init__(self):
        self.camera_ids_server_ips = {}
        self.camera_config = config['roborpc']['cameras']
        self.composed_multi_cameras = {}

    def connect_now(self):
        server_ips_address = self.camera_config['server_ips_address']
        server_rpc_ports = self.camera_config['server_rpc_ports']
        for server_ip_address, server_rpc_port in zip(server_ips_address, server_rpc_ports):
            if server_rpc_port == "":
                self.composed_multi_cameras[server_ip_address] = MultiCameras()
            else:
                self.composed_multi_cameras[server_ip_address] = MultiCamerasRpc(server_ip_address, server_rpc_port)
            self.composed_multi_cameras[server_ip_address].connect_now()
            logger.info(f"MultiCamerasRpc connected to {server_ip_address}:{server_rpc_port}")

    def disconnect_now(self):
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            multi_camera.disconnect_now()
            logger.info(f"MultiCamerasRpc disconnected from {server_ip_address}")

    def get_device_ids(self) -> List[str]:
        camera_ids = []
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            camera_ids.extend(multi_camera.get_device_ids())
        return camera_ids

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        camera_intrinsics = {}
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            camera_intrinsics[server_ip_address] = multi_camera.get_camera_intrinsics()
        return camera_intrinsics

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        camera_extrinsics = {}
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            camera_extrinsics[server_ip_address] = multi_camera.get_camera_extrinsics()
        return camera_extrinsics

    def read_camera(self) -> Dict[str, Dict[str, bytes]]:
        camera_info = {}
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            camera_info[server_ip_address] = multi_camera.read_camera()
        new_camera_info = {}
        for server_ip_address, camera_info in camera_info.items():
            for camera_id, info in camera_info.items():
                new_camera_info[camera_id] = {'color': pickle.loads(info['color']),
                                              'depth': pickle.loads(info['depth'])}
        return new_camera_info
