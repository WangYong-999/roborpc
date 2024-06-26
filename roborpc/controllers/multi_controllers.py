from typing import Union, List, Dict

import numpy as np

from roborpc.controllers.controller_base import ControllerBase
from roborpc.controllers.dynamixel_controller import DynamixelController
from roborpc.robots.dynamixel import Dynamixel
from roborpc.controllers.dynamixel_controller_utils import get_joint_offsets
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger


class MultiControllers(ControllerBase):
    def __init__(self):
        super().__init__()
        self.controllers = {}
        self.controller_config = config['roborpc']['controllers']
        self.robot_config = config['roborpc']['robots']
        self.controller_ids = self.controller_config['controller_ids'][0]

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        result = {}
        for controller_id in self.controller_ids:
            if str(controller_id).startswith('dynamixel_controller'):
                joint_offset_list, gripper_offset_list = get_joint_offsets(
                    joint_ids=self.robot_config['dynamixel'][controller_id]["joint_ids"],
                    joint_signs=self.robot_config['dynamixel'][controller_id]["joint_signs"],
                    start_joints=np.array(self.robot_config['dynamixel'][controller_id]["start_joints"]),
                    port=self.robot_config['dynamixel'][controller_id]["port"],
                    gripper_config=self.robot_config['dynamixel'][controller_id]["gripper_config"],
                    baudrate=self.robot_config['dynamixel'][controller_id]["baudrate"]
                )
                gripper_config = (len(joint_offset_list), gripper_offset_list[0], gripper_offset_list[1])
                self.controllers[controller_id] = DynamixelController(
                    Dynamixel(
                        robot_id=controller_id,
                        joint_ids=self.robot_config['dynamixel'][controller_id]["joint_ids"],
                        joint_signs=self.robot_config['dynamixel'][controller_id]["joint_signs"],
                        joint_offsets=joint_offset_list,
                        start_joints=np.array(self.robot_config['dynamixel'][controller_id]["start_joints"]),
                        port=self.robot_config['dynamixel'][controller_id]["port"],
                        gripper_config=gripper_config,
                        baudrate=self.robot_config['dynamixel'][controller_id]["baudrate"]
                    )
                )
                result[controller_id] = self.controllers[controller_id].connect_now()
            else:
                logger.error(f"Controller {controller_id} not found in config file.")
            logger.info(f"Controller {controller_id} connected successfully.")
            return result

    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        result = {}
        for controller_id in self.controller_ids:
            result[controller_id] = self.controllers[controller_id].disconnect_now()
            logger.info(f"Controller {controller_id} disconnected successfully.")
        return result

    def get_controller_id(self) -> List[str]:
        return self.controller_ids

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        info_dict = {}
        for controller_id in self.controller_ids:
            info_dict[controller_id] = self.controllers[controller_id].get_info()
        return info_dict

    def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        result_dict = {}
        for controller_id in self.controller_ids:
            result_dict[controller_id] = self.controllers[controller_id].forward(obs_dict[controller_id])
        return result_dict


if __name__ == '__main__':
    import zerorpc

    multi_controller = MultiControllers()
    s = zerorpc.Server(multi_controller)
    s.bind(f"tcp://0.0.0.0:{multi_controller.controller_config['sever_rpc_ports'][0]}")
    s.run()
