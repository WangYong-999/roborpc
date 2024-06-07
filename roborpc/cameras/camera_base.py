
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class CameraBase(ABC):

    @abstractmethod
    def connect(self):
        """Connect to the camera."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        """Disconnect from the camera."""
        raise NotImplementedError

    @abstractmethod
    def get_device_ids(self) -> List[str]:
        """Get the device ids of the available cameras."""
        pass

    @abstractmethod
    def get_camera_intrinsics(self) -> Dict[str, List[float]]:
        """
        Get the intrinsics of the camera.
        """
        raise NotImplementedError

    @abstractmethod
    def get_camera_extrinsics(self) -> Dict[str, List[float]]:
        """
        Get the extrinsics of the camera.
        """
        raise NotImplementedError

    @abstractmethod
    def read_camera(self) -> Dict[str, Dict[str, str]]:
        """Read a frame from the camera.

        Returns:
            np.ndarray: The color image.
            np.ndarray: The depth image.
        """
        raise NotImplementedError




