import subprocess
import traceback
import time
import numpy as np
from roborpc.common.logger_loader import logger
from roborpc.kinematics_solver.trajectory_interpolation import action_linear_interpolation
from roborpc.robot_env import RobotEnv

if __name__ == '__main__':
    try:
        robot_env = RobotEnv()
        controller = robot_env.controllers
        while True:
            try:
                robot_obs, camera_obs = robot_env.get_observation()
                action = controller.forward(robot_obs)
                robot_env.step(action)
            except Exception as e:
                logger.error(f"traceback: {traceback.format_exc()}")
                pid = subprocess.run(["pgrep", "-f", "test_controller"], capture_output=True)
                subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                break
    except Exception as e:
        logger.error(f"traceback: {traceback.format_exc()}")
        pid = subprocess.run(["pgrep", "-f", "test_controller"], capture_output=True)
        subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
