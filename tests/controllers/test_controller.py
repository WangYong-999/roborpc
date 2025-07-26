import subprocess
import sys
from roborpc.kinematics_solver.trajectory_interpolation import action_linear_interpolation
from roborpc.robot_env import RobotEnv

if __name__ == '__main__':
    robot_env = RobotEnv()
    controller = robot_env.controllers

    # move leader position
    move_leader_pos = False


    while True:
        try:
            if not move_leader_pos:
                robot_env.reset()
                robot_obs, camera_obs = robot_env.get_observation()
                action = controller.forward(robot_obs)
                robot_env.step(action, blocking=True)
                move_leader_pos = True

            robot_obs, camera_obs = robot_env.get_observation()
            # print(robot_obs)
            action = controller.forward(robot_obs)
            # print(action)
            # action = action_linear_interpolation(robot_obs, action)
            robot_env.step(action)
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            pid = subprocess.run(["pgrep", "-f", "test_controller"], capture_output=True)
            subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
            sys.exit(0)


