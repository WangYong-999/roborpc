from typing import Union, List, Dict

from roborpc.controllers.controller_base import ControllerBase
from roborpc.controllers.dynamixel_controller import DynamixelController
from roborpc.robots.dynamixel import Dynamixel
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger


class MultiControllers(ControllerBase):
    def __init__(self):
        super().__init__()
        self.controllers = {}
        self.controller_config = config['roborpc']['controllers']
        self.controller_ids = self.controller_config['controller_ids'][0]

    def connect_now(self):
        for controller_id in self.controller_ids:
            if str(controller_id).startswith('dynamixel_controller'):
                self.controllers[controller_id] = DynamixelController(
                    Dynamixel(
                        robot_id=controller_id,
                        joint_ids=self.controller_config['dynamixel'][controller_id]["joint_ids"],
                        joint_signs=self.controller_config['dynamixel'][controller_id]["joint_signs"],
                        joint_offsets=self.controller_config['dynamixel'][controller_id]["joint_offsets"],
                        start_joints=self.controller_config['dynamixel'][controller_id]["start_joints"],
                        port=self.controller_config['dynamixel'][controller_id]["port"],
                        gripper_config=self.controller_config['dynamixel'][controller_id]["gripper_config"]
                    )
                )
                self.controllers[controller_id].connect_now()
            else:
                logger.error(f"Controller {controller_id} not found in config file.")
            logger.info(f"Controller {controller_id} connected successfully.")

    def disconnect_now(self):
        for controller_id in self.controller_ids:
            self.controllers[controller_id].disconnect_now()
            logger.info(f"Controller {controller_id} disconnected successfully.")

    def get_controller_id(self) -> List[str]:
        return self.controller_ids

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        info_dict = {}
        for controller_id in self.controller_ids:
            info_dict[controller_id] = self.controllers[controller_id].get_info()
        return info_dict

    def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]):
        for controller_id in self.controller_ids:
            self.controllers[controller_id].forward(obs_dict[controller_id])


if __name__ == '__main__':
    import zerorpc

    multi_controller = MultiControllers()
    s = zerorpc.Server(multi_controller)
    s.bind(f"tcp://0.0.0.0:{multi_controller.controller_config['server_port'][0]}")
    s.run()
