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

    def connect_now(self):
        self.controllers = zerorpc.Client(heartbeat=20)
        self.controllers.connect("tcp://" + self.server_ip_address + ":" + self.rpc_port)
        self.controllers.connect_now()

    def disconnect_now(self):
        self.controllers.disconnect_now()
        self.controllers.close()

    def get_controllers(self) -> List[str]:
        return self.controllers.get_controllers()

    async def get_info(self) -> Union[Dict[str, Dict[str, bool]], Dict[str, bool]]:
        return self.controllers.get_info()

    async def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]):
        self.controllers.forward(obs_dict)


class ComposedMultiController(ControllerBase):
    def __init__(self):
        super().__init__()
        self.composed_multi_controllers = {}
        self.controller_config = config['roborpc']['controllers']
        self.controller_ids_server_ips = {}
        self.loop = asyncio.get_event_loop()

    def connect_now(self):
        server_ips_address = self.controller_config["server_ips_address"]
        sever_rpc_ports = self.controller_config["sever_rpc_ports"]
        for server_ip_address, rpc_port in zip(server_ips_address, sever_rpc_ports):
            self.composed_multi_controllers[server_ip_address] = MultiControllersRpc(server_ip_address, rpc_port)
            self.composed_multi_controllers[server_ip_address].connect_now()
            logger.info("Connected to server: " + server_ip_address + ":" + rpc_port)
        self.controller_ids_server_ips = self.get_controller_ids_server_ips()

    def disconnect_now(self):
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            multi_controllers.disconnect_now()
            logger.info("Disconnected from server: " + server_ip_address)

    def get_controller_ids_server_ips(self) -> Dict[str, str]:
        controller_ids_server_ips = {}
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            for controller_ids in multi_controllers.get_controller_ids():
                for controller_id in controller_ids:
                    controller_ids_server_ips[controller_id] = server_ip_address
        return controller_ids_server_ips

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

    def forward(self, obs_dict: Union[List[float], Dict[str, List[float]]]):
        multi_controllers_task = []
        for server_ip_address, multi_controllers in self.composed_multi_controllers.items():
            new_obs_dict = {}
            for controller_id in multi_controllers.get_controller_ids():
                new_obs_dict[controller_id].update(obs_dict[controller_id])
            multi_controllers_task.append(multi_controllers.forward(new_obs_dict))
        self.loop.run_until_complete(asyncio.gather(*multi_controllers_task))

