import threading
import gym
from roborpc.common.config_loader import config

from roborpc.robots.realman import MultiRealMan as Robot
from roborpc.robots.isaac_sim_franka_rpc import MultiIsaacSimFrankaRpc
from roborpc.cameras.realsense_camera import MultiRealSenseCamera


class RobotEnv(gym.Env):
    _single_lock = threading.Lock()
    _instance = None

    def __new__(cls):
        # single instance
        with RobotEnv._single_lock:
            RobotEnv._instance = object.__new__(cls)
            RobotEnv._instance._initialize()
            return RobotEnv._instance

    def _initialize(self):
        self.robot = MultiIsaacSimFrankaRpc()
        self.camera = MultiRealSenseCamera()

    def __del__(self):
        self.robot.disconnect()
        self.camera.disconnect()



