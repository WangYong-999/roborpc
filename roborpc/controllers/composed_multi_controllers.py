import asyncio
from typing import Union, List, Dict
import zerorpc

from roborpc.controllers.controller_base import ControllerBase
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.controllers.multi_controllers import MultiControllers


class MultiControllersRpc(ControllerBase):
    def __init__(self, server_ip_address: str, rpc_port: str):
        super().__init__()
        self.server_ip_address = server_ip_address
        self.rpc_port = rpc_port
        self.controllers = None

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        self.controllers = zerorpc.Client(heartbeat=20)
        self.controllers.connect("tcp://" + self.server_ip_address + ":" + self.rpc_port)
        return self.controllers.connect_now()

    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        result = self.controllers.disconnect_now()
        self.controllers.close()
        return result

    def get_controllers(self) -> List[str]:
        return self.controllers.get_controllers()

    def get_controller_id(self) -> List[str]:
        return self.controllers.get_controller_id()

    def get_control_robot_map(self) -> Dict[str, str]:
        return self.controllers.get_control_robot_map()

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        return self.controllers.get_info()

    def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[
        Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        return self.controllers.forward(obs_dict)


class ComposedMultiController(ControllerBase):

    def __init__(self, kinematic_solver = None):
        super().__init__()
        self.kinematic_solver = kinematic_solver
        self.composed_multi_controllers = {}
        self.controller_config = config['roborpc']['controllers']
        self.robot_controller_map = {}

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        result = {}
        server_ips_address = self.controller_config["server_ips_address"]
        sever_rpc_ports = self.controller_config["sever_rpc_ports"]
        for server_ip_address, rpc_port in zip(server_ips_address, sever_rpc_ports):
            if rpc_port == "":
                self.composed_multi_controllers[server_ip_address] = MultiControllers(kinematic_solver=self.kinematic_solver)
            else:
                self.composed_multi_controllers[server_ip_address] = MultiControllersRpc(server_ip_address, rpc_port)
            result.update(self.composed_multi_controllers[server_ip_address].connect_now())
            self.robot_controller_map[server_ip_address] = self.composed_multi_controllers[
                server_ip_address].get_control_robot_map()
            logger.info("Connected to server: " + server_ip_address + ":" + rpc_port)
        return result

    def disconnect_now(self) -> Union[bool, Dict[str, bool]]:
        result = {}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            result.update(multi_controllers.disconnect_now())
            logger.info("Disconnected from server: " + server_ip_address)
        return result

    def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        info_dict = {}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            info_dict[server_ip_address] = multi_controllers.get_info()
        new_info_dict = {}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            robot_info_dict = info_dict[server_ip_address]
            for controller_id, controller_info in robot_info_dict.items():
                new_info_dict[controller_id] = controller_info
        return new_info_dict

    def get_controller_id(self) -> List[str]:
        controller_ids = []
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            controller_ids.extend(multi_controllers.get_controller_id())
        return controller_ids

    def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[
        Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        result_dict = {}
        new_obs_dict = {}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            robot_controller_map = self.robot_controller_map[server_ip_address]
            for controller_id, robot_id in robot_controller_map.items():
                new_obs_dict[server_ip_address] = {robot_id: obs_dict[robot_id]}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            result_dict.update(multi_controllers.forward(new_obs_dict[server_ip_address]))
        return result_dict
