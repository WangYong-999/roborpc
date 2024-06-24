from abc import ABC, abstractmethod
from typing import Dict, List, Union


class ControllerBase(ABC):

    @abstractmethod
    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        """
        Connect to the controller.
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        """
        Disconnect from the controller.
        """
        raise NotImplementedError

    @abstractmethod
    def get_controller_id(self) -> List[str]:
        """
        Get the id of the controller
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        """
        Get information about the controller.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]) -> Union[List[float], Dict[str, List[float]]]:
        """
        Move the robot forward.
        """
        raise NotImplementedError

