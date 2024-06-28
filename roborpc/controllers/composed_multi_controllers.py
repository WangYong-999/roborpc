import asyncio
from typing import Union, List, Dict
import zerorpc

from roborpc.controllers.controller_base import ControllerBase
from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger


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

    async def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        return self.controllers.get_info()

    async def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        return self.controllers.forward(obs_dict)


class ComposedMultiController(ControllerBase):


    def __init__(self):
        super().__init__()
        self.composed_multi_controllers = {}
        self.controller_config = config['roborpc']['controllers']
        self.loop = asyncio.get_event_loop()

    def connect_now(self) -> Union[bool, Dict[str, bool]]:
        result = {}
        server_ips_address = self.controller_config["server_ips_address"]
        sever_rpc_ports = self.controller_config["sever_rpc_ports"]
        for server_ip_address, rpc_port in zip(server_ips_address, sever_rpc_ports):
            self.composed_multi_controllers[server_ip_address] = MultiControllersRpc(server_ip_address, rpc_port)
            result.update(self.composed_multi_controllers[server_ip_address].connect_now())
            logger.info("Connected to server: " + server_ip_address + ":" + rpc_port)
            print(result)
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
            info_dict[server_ip_address] = asyncio.ensure_future(multi_controllers.get_info())
        self.loop.run_until_complete(asyncio.gather(*info_dict.values()))
        new_info_dict = {}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            robot_info_dict = info_dict[server_ip_address].result()
            for controller_id, controller_info in robot_info_dict.items():
                new_info_dict[controller_id] = controller_info
        return new_info_dict

    def get_controller_id(self) -> List[str]:
        controller_ids = []
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            controller_ids.extend(multi_controllers.get_controller_id())
        return controller_ids

    def forward(self, obs_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        result_dict = {}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            new_obs_dict = {}
            for controller_id in multi_controllers.get_controller_id():
                new_obs_dict.update({controller_id: obs_dict[controller_id]})
            result_dict[server_ip_address] = asyncio.ensure_future(multi_controllers.forward(new_obs_dict[server_ip_address]))
        self.loop.run_until_complete(asyncio.gather(*result_dict.values()))
        new_result_dict = {}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            robot_result_dict = result_dict[server_ip_address].result()
            for controller_id, controller_result in robot_result_dict.items():
                new_result_dict[controller_id] = controller_result
        return new_result_dict

