from typing import Union, List, Dict

from roborpc.robots.robot_base import RobotBase

from roborpc.sim_robots.isaac_sim.single_franka_vision import SingleFrankaVision as Robot


class SimRobotInterface(RobotBase):
    def __init__(self):
        self.robot = Robot()

    def connect(self):
        self.robot.connect()

    def disconnect(self):
        pass

    def get_robot_ids(self) -> List[str]:
        pass

    def set_ee_pose(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "cartesian_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def set_joints(self, action: Union[List[float], Dict[str, List[float]]],
                   action_space: Union[str, List[str]] = "joint_position", blocking: Union[bool, List[bool]] = False):
        pass

    def set_gripper(self, action: Union[List[float], Dict[str, List[float]]],
                    action_space: Union[str, List[str]] = "gripper_position",
                    blocking: Union[bool, List[bool]] = False):
        pass

    def get_robot_state(self) -> Dict[str, List[float]]:
        pass

    def get_dofs(self) -> Dict[str, int]:
        pass

    def get_joint_positions(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_gripper_position(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_joint_velocities(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def get_ee_pose(self) -> Union[List[float], Dict[str, List[float]]]:
        pass

    def __init__(self, name, ip_address, port):
        super().__init__(name, ip_address, port)


if __name__ == '__main__':
    import zerorpc
    sim_robot = SimRobotInterface()
    s = zerorpc.Server(sim_robot)
    s.bind(f"tcp://0.0.0.0:{sim_robot.robot_config['rpc_port'][0]}")
    s.run()
