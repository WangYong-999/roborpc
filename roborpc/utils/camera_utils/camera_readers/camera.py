import abc


class CameraDriver(abc.ABC):
    """
    For a camera driver. This is used to abstract the camera from the rest of the code.
    """

    def read_camera(self):
        """Read a frame from the camera.
        """
        raise NotImplementedError

    def start_recording(self, filename):
        """
        Start recording to a file.
        Args:
            filename: record to this file.

        """
        raise NotImplementedError

    def stop_recording(self):
        """
        Stop recording.
        """
        raise NotImplementedError

    def disable_camera(self):
        """
        Disable the camera.
        """
        raise NotImplementedError

    def is_running(self):
        """
        Check if the camera is running.
        Returns:
        """
        raise NotImplementedError

    def get_intrinsics(self):
        """
        Get the camera intrinsics.
        Returns:
        """
        raise NotImplementedError

    def set_calibration_mode(self):
        """
        Set the camera to calibration mode.
        """
        raise NotImplementedError

    def set_trajectory_mode(self):
        """
        Set the camera to trajectory mode.
        """
        raise NotImplementedError

    def set_reading_parameters(self):
        """
        Set the camera to reading mode.
        """
        raise NotImplementedError

    def enable_advanced_calibration(self):
        """
        Enable advanced calibration mode.
        """
        raise NotImplementedError

    def disable_advanced_calibration(self):
        """
        Disable advanced calibration mode.
        """
        raise NotImplementedError


