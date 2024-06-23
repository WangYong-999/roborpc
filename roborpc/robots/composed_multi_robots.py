import asyncio
from typing import Union, List, Dict

import zerorpc

from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.robots.robot_base import RobotBase


class MutilRobotsRpc(RobotBase):

    def __init__(self, server_ip_address: str, rpc_port: str):
        super().__init__()
        self.server_ip_address = server_ip_address
        self.rpc_port = rpc_port
        self.robots = None

    def connect_now(self):
        self.robots = zerorpc.Client(heartbeat=20)
        self.robots.connect("tcp://" + self.server_ip_address + ":" + self.rpc_port)
        self.robots.connect_now()

    def disconnect_now(self):
        self.robots.disconnect_now()
        self.robots.close()

    def get_robot_ids(self) -> List[str]:
        return self.robots.get_robot_ids()

    async def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                          action_space: Union[str, Dict[str, str]] = "cartesian_position",
                          blocking: Union[bool, Dict[str, bool]] = False):
        self.robots.set_ee_pose(action, action_space, blocking)

    async def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                         action_space: Union[str, Dict[str, str]] = "joint_position",
                         blocking: Union[bool, Dict[str, bool]] = False):
        self.robots.set_joints(action, action_space, blocking)

    async def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                          action_space: Union[str, Dict[str, str]] = "gripper_position",
                          blocking: Union[bool, Dict[str, bool]] = False):
        self.robots.set_gripper(action, action_space, blocking)

    async def get_robot_state(self) -> Dict[str, List[float]]:
        return self.robots.get_robot_state()

    def get_dofs(self) -> Dict[str, int]:
        return self.robots.get_dofs()

    async def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robots.get_joint_positions()

    async def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robots.get_gripper_position()

    async def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robots.get_joint_velocities()

    async def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        return self.robots.get_ee_pose()


class ComposedMultiRobots(RobotBase):
    def __init__(self):
        self.robot_ids_server_ips = {}
        self.robot_config = config["roborpc"]["robots"]
        self.composed_multi_robots = {}
        self.loop = asyncio.get_event_loop()

    def connect_now(self):
        server_ips_address = self.robot_config["server_ips_address"]
        sever_rpc_ports = self.robot_config["sever_rpc_ports"]
        for server_ip_address, sever_rpc_port in zip(server_ips_address, sever_rpc_ports):
            self.composed_multi_robots[server_ip_address] = MutilRobotsRpc(server_ip_address, sever_rpc_port)
            self.composed_multi_robots[server_ip_address].connect_now()
            logger.info(f"Multi Robot {server_ip_address}:{sever_rpc_port} Connect Success!")
        self.robot_ids_server_ips = self.get_robot_ids_server_ips()

    def disconnect_now(self):
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            multi_robots.disconnect_now()
            logger.info(f"Multi Robot {server_ip_address} Disconnect Success!")

    def get_robot_ids_server_ips(self) -> Dict[str, str]:
        robot_ids_server_ips = {}
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            for robot_ids in multi_robots.get_robot_ids():
                for robot_id in robot_ids:
                    robot_ids_server_ips[robot_id] = server_ip_address
        return robot_ids_server_ips

    def get_robot_ids(self) -> List[str]:
        robot_ids = []
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            robot_ids.extend(multi_robots.get_robot_ids())
        return robot_ids

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "cartesian_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        multi_robots_task = []
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            new_action = {}
            new_action_space = {}
            new_blocking = {}
            for robot_id in multi_robots.get_robot_ids():
                new_action.update({robot_id: action[robot_id]})
                new_action_space.update({robot_id: action_space[robot_id]})
                new_blocking.update({robot_id: blocking[robot_id]})
            multi_robots_task.append(multi_robots.set_ee_pose(new_action, new_action_space, new_blocking))
        self.loop.run_until_complete(asyncio.gather(*multi_robots_task))

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, Dict[str, str]] = "joint_position",
                   blocking: Union[bool, Dict[str, bool]] = False):
        multi_robots_task = []
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            new_action = {}
            new_action_space = {}
            new_blocking = {}
            for robot_id in multi_robots.get_robot_ids():
                new_action.update({robot_id: action[robot_id]})
                new_action_space.update({robot_id: action_space[robot_id]})
                new_blocking.update({robot_id: blocking[robot_id]})
            multi_robots_task.append(multi_robots.set_joints(new_action, new_action_space, new_blocking))
        self.loop.run_until_complete(asyncio.gather(*multi_robots_task))

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, Dict[str, str]] = "gripper_position",
                    blocking: Union[bool, Dict[str, bool]] = False):
        multi_robots_task = []
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            new_action = {}
            new_action_space = {}
            new_blocking = {}
            for robot_id in multi_robots.get_robot_ids():
                new_action.update({robot_id: action[robot_id]})
                new_action_space.update({robot_id: action_space[robot_id]})
                new_blocking.update({robot_id: blocking[robot_id]})
            multi_robots_task.append(multi_robots.set_gripper(new_action, new_action_space, new_blocking))
        self.loop.run_until_complete(asyncio.gather(*multi_robots_task))

    def get_robot_state(self) -> Dict[str, List[float]]:
        robot_states = {}
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            robot_states[server_ip_address] = asyncio.ensure_future(multi_robots.get_robot_state())
        self.loop.run_until_complete(asyncio.gather(*robot_states.values()))
        new_robot_state = {}
        for server_ip_address, robot_state in robot_states.items():
            robot_states[server_ip_address] = robot_state.result()
            for robot_id, state in robot_states[server_ip_address].items():
                new_robot_state[robot_id] = state
        return new_robot_state

    def get_dofs(self) -> Dict[str, int]:
        dofs = {}
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            dofs[server_ip_address] = multi_robots.get_dofs()
        new_dofs = {}
        for server_ip_address, robot_dofs in dofs.items():
            for robot_id, dof in robot_dofs.items():
                new_dofs[robot_id] = dof
        return new_dofs

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        joint_positions = {}
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            joint_positions[server_ip_address] = asyncio.ensure_future(multi_robots.get_joint_positions())
        self.loop.run_until_complete(asyncio.gather(*joint_positions.values()))
        new_joint_positions = {}
        for server_ip_address, joint_position in joint_positions.items():
            joint_positions[server_ip_address] = joint_position.result()
            for robot_id, position in joint_positions[server_ip_address].items():
                new_joint_positions[robot_id] = position
        return new_joint_positions

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        gripper_positions = {}
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            gripper_positions[server_ip_address] = asyncio.ensure_future(multi_robots.get_gripper_position())
        self.loop.run_until_complete(asyncio.gather(*gripper_positions.values()))
        new_gripper_positions = {}
        for server_ip_address, gripper_position in gripper_positions.items():
            gripper_positions[server_ip_address] = gripper_position.result()
            for robot_id, position in gripper_positions[server_ip_address].items():
                new_gripper_positions[robot_id] = position
        return new_gripper_positions

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        joint_velocities = {}
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            joint_velocities[server_ip_address] = asyncio.ensure_future(multi_robots.get_joint_velocities())
        self.loop.run_until_complete(asyncio.gather(*joint_velocities.values()))
        new_joint_velocities = {}
        for server_ip_address, joint_velocity in joint_velocities.items():
            joint_velocities[server_ip_address] = joint_velocity.result()
            for robot_id, velocity in joint_velocities[server_ip_address].items():
                new_joint_velocities[robot_id] = velocity
        return new_joint_velocities

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        ee_poses = {}
        for server_ip_address, multi_robots in self.composed_multi_robots.items():
            ee_poses[server_ip_address] = asyncio.ensure_future(multi_robots.get_ee_pose())
        self.loop.run_until_complete(asyncio.gather(*ee_poses.values()))
        new_ee_poses = {}
        for server_ip_address, ee_pose in ee_poses.items():
            ee_poses[server_ip_address] = ee_pose.result()
            for robot_id, pose in ee_poses[server_ip_address].items():
                new_ee_poses[robot_id] = pose
        return new_ee_poses


if __name__ == '__main__':
    import zerorpc
    multi_realman = ComposedMultiRobots()
    multi_realman.connect_now()
    print(multi_realman.get_robot_ids())
    print(multi_realman.get_robot_state())

    multi_realman.set_joints({"realman_1": [0.10136872295583066, 0.059864793343405505, -0.14184290830957919, -1.8463838156848014,
                              0.01965240737745615, -0.2019695010407838, 0.364869513188684]},
                             action_space={"realman_1": "joint_position"}, blocking={"realman_1": True})

    multi_realman.disconnect_now()
