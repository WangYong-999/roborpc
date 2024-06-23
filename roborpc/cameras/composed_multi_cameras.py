import asyncio
from typing import Dict, List
import zerorpc

from roborpc.cameras.camera_base import CameraBase
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger


class MultiCamerasRpc(CameraBase):
    def __init__(self, server_ip_address: str, server_port: int):
        super().__init__()
        self.server_ip_address = server_ip_address
        self.server_port = server_port
        self.cameras = None

    def connect_now(self):
        self.cameras = zerorpc.Client(heartbeat=20)
        self.cameras.connect("tcp://{}:{}".format(self.server_ip_address, self.server_port))
        self.cameras.connect_now()

    def disconnect_now(self):
        self.cameras.disconnect_now()
        self.cameras.close()

    def get_device_ids(self) -> List[str]:
        return  self.cameras.get_device_ids()

    async def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        return self.cameras.get_camera_intrinsics()

    async def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        return self.cameras.get_camera_extrinsices()

    async def read_camera(self) -> Dict[str, Dict[str, str]]:
        return self.cameras.read_camera()


class ComposedMultiCameras(CameraBase):
    def __init__(self):
        self.camera_ids_server_ips = {}
        self.camera_config = config['roborpc']['cameras']
        self.composed_multi_cameras = {}
        self.loop = asyncio.get_event_loop()
    
    def connect_now(self):
        server_ips_address = config['roborpc']['server_ips_address']
        server_rpc_ports = config['roborpc']['server_rpc_ports']
        for server_ip_address, server_rpc_port in zip(server_ips_address, server_rpc_ports):
            self.camera_ids_server_ips[server_ip_address] = server_rpc_port
            self.composed_multi_cameras[server_ip_address] = MultiCamerasRpc(server_ip_address, server_rpc_port)
            self.composed_multi_cameras[server_ip_address].connect_now()
            logger.info(f"MultiCamerasRpc connected to {server_ip_address}:{server_rpc_port}")
        self.camera_ids_server_ips = self.get_camera_ids_server_ips()

    def disconnect_now(self):
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            multi_camera.disconnect_now()
            logger.info(f"MultiCamerasRpc disconnected from {server_ip_address}")
    
    def get_camera_ids_server_ips(self) -> Dict[str, Dict[str, str]]:
        camera_ids_server_ips = {}
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            for camera_ids in multi_camera.get_device_ids():
                for camera_id in camera_ids:
                    camera_ids_server_ips[camera_id] = server_ip_address
        return camera_ids_server_ips

    def get_device_ids(self) -> List[str]:
        camera_ids = []
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            camera_ids.extend(multi_camera.get_device_ids())
        return camera_ids

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        camera_intrinsics = {}
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            camera_intrinsics[server_ip_address] = asyncio.ensure_future(multi_camera.get_camera_intrinsics())
        self.loop.run_until_complete(asyncio.gather(*camera_intrinsics.values()))
        new_camera_intrinsics = {}
        for server_ip_address, intrinsics in camera_intrinsics.items():
            camera_intrinsics[server_ip_address] = intrinsics.result()
            for camera_id, intrinsic in camera_intrinsics[server_ip_address].items():
                new_camera_intrinsics[camera_id] = intrinsic
        return new_camera_intrinsics

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        camera_extrinsics = {}
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            camera_extrinsics[server_ip_address] = asyncio.ensure_future(multi_camera.get_camera_extrinsics())
        self.loop.run_until_complete(asyncio.gather(*camera_extrinsics.values()))
        new_camera_extrinsics = {}
        for server_ip_address, extrinsics in camera_extrinsics.items():
            camera_extrinsics[server_ip_address] = extrinsics.result()
            for camera_id, extrinsic in camera_extrinsics[server_ip_address].items():
                new_camera_extrinsics[camera_id] = extrinsic
        return new_camera_extrinsics

    def read_camera(self) -> Dict[str, Dict[str, str]]:
        camera_info = {}
        for server_ip_address, multi_camera in self.composed_multi_cameras.items():
            camera_info[server_ip_address] = asyncio.ensure_future(multi_camera.read_camera())
        self.loop.run_until_complete(asyncio.gather(*camera_info.values()))
        new_camera_info = {}
        for server_ip_address, camera_info in camera_info.items():
            camera_info[server_ip_address] = camera_info.result()
            for camera_id, info in camera_info[server_ip_address].items():
                new_camera_info[camera_id] = info
        return new_camera_info
    