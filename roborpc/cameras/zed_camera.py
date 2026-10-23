from typing import Dict, List

from roborpc.cameras.camera_base import CameraBase


class ZedCamera(CameraBase):
    def connect_now(self):
        pass

    def disconnect_now(self):
        pass

    def get_device_ids(self) -> List[str]:
        pass

    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        pass

    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        pass

    def read_camera(self) -> Dict[str, Dict[str, bytes]]:
        pass
