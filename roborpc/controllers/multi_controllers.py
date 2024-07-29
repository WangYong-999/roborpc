from typing import Union, List, Dict

import numpy as np

from roborpc.controllers.controller_base import ControllerBase
from roborpc.controllers.dynamixel_controller import DynamixelController
from roborpc.controllers.keyboard_controller import KeyboardController
from roborpc.controllers.quest_controller import QuestController
from roborpc.controllers.spacemouse_controller import SpaceMouseController
from roborpc.robots.dynamixel import Dynamixel
from roborpc.controllers.dynamixel_controller_utils import get_joint_offsets
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger


class MultiControllers(ControllerBase):
    def __init__(self, controller_ids: List[str] = None, control_robot_ids: List[str] = None,
                 kinematic_solver=None):
        super().__init__()
        self.controllers = {}
        self.kinematic_solver = kinematic_solver
        self.controller_config = config['roborpc']['controllers']
        self.robot_config = config['roborpc']['robots']
        if controller_ids is not None:
            self.controller_ids = controller_ids
        else:
            self.controller_ids = self.controller_config['controller_ids'][0]
        if control_robot_ids is not None:
            self.control_robot_ids = control_robot_ids
        else:
            self.control_robot_ids = self.controller_config['control_robot_ids'][0]

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
            elif str(controller_id).startswith('spacemouse_controller'):
                self.controllers[controller_id] = SpaceMouseController(kinematic_solver=self.kinematic_solver)
                result[controller_id] = self.controllers[controller_id].connect_now()
            elif str(controller_id).startswith('quest_controller'):
                self.controllers[controller_id] = QuestController(kinematic_solver=self.kinematic_solver)
                result[controller_id] = self.controllers[controller_id].connect_now()
            elif str(controller_id).startswith('keyboard_controller'):
                self.controllers[controller_id] = KeyboardController(kinematic_solver=self.kinematic_solver)
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

    def get_control_robot_map(self) -> Dict[str, str]:
        return {controller_id: control_robot_id for controller_id, control_robot_id
                in zip(self.controller_ids, self.control_robot_ids)}

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        info_dict = {}
        for controller_id in self.controller_ids:
            info_dict[controller_id] = self.controllers[controller_id].get_info()
        return info_dict

    def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        result_dict = {}
        for idx, controller_id in enumerate(self.controller_ids):
            result_dict[self.control_robot_ids[idx]] = self.controllers[controller_id].forward(obs_dict[self.control_robot_ids[idx]])
        return result_dict


if __name__ == '__main__':
    import zerorpc
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--controller_ids', type=str, nargs='+', default=None, help='controller ids to connect')
    parser.add_argument('--control_robot_ids', type=str, nargs='+', default=None, help='robot ids to control')
    args = parser.parse_args()

    controller_ids = args.controller_ids
    control_robot_ids = args.control_robot_ids

    multi_controller = MultiControllers(controller_ids=list(controller_ids.split(',')) if controller_ids is not None else None,
                                        control_robot_ids=list(control_robot_ids.split(',')) if control_robot_ids is not None else None)
    s = zerorpc.Server(multi_controller)
    rpc_port = multi_controller.controller_config['sever_rpc_ports'][0]
    logger.info(f"RPC server started on port {rpc_port}")
    s.bind(f"tcp://0.0.0.0:{rpc_port}")
    s.run()
