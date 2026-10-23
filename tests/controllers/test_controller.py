import subprocess
import sys
from roborpc.kinematics_solver.trajectory_interpolation import action_linear_interpolation
from roborpc.robot_env import RobotEnv

if __name__ == '__main__':
    robot_env = RobotEnv()
    controller = robot_env.controllers

    while True:
        try:
            robot_obs, camera_obs = robot_env.get_observation()
            # print(camera_obs)
            # print(robot_obs)
            action = controller.forward(robot_obs)
            result = action_linear_interpolation(robot_obs, action)
            print(result)
            robot_env.step(result)
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            pid = subprocess.run(["pgrep", "-f", "test_controller"], capture_output=True)
            subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
            sys.exit(0)


