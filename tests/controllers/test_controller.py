import os
import subprocess

from roborpc.common.config_loader import config
from roborpc.robot_env import RobotEnv
from roborpc.controllers.composed_multi_controllers import ComposedMultiController

if __name__ == '__main__':
    try:
        pid = subprocess.Popen('bash -c "python /home/jz08/code_repo/roborpc/roborpc/controllers/multi_controllers.py"',
                               shell=True)
        controller = ComposedMultiController()
        result = controller.connect_now()
        robot_env = RobotEnv()
        print(result)
        for r in result.values():
            if not r:
                raise Exception("Failed to connect to all controllers")
        while True:
            try:
                obs = robot_env.get_observation()
                action = controller.forward(obs)
                print(action)
                # robot_env.step(action)
            except KeyboardInterrupt:
                pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
                subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                break
    except Exception as e:
        print(e)
        pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
        subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
