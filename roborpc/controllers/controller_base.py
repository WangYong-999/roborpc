from abc import ABC, abstractmethod
from typing import Dict, List, Union


class ControllerBase(ABC):

    @abstractmethod
    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        pass

    @abstractmethod
    def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]):
        pass

