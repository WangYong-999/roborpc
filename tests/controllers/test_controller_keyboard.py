import subprocess
import sys

import numpy as np

from roborpc.kinematics_solver.trajectory_interpolation import action_linear_interpolation
from roborpc.robot_env import RobotEnv

if __name__ == '__main__':
    robot_env = RobotEnv()
    controller = robot_env.controllers

    # move leader position
    move_leader_pos = False

    # save robot state
    num_steps = 500
    robot_obs_list = []
    # telep
    while True:
        try:
            if not move_leader_pos:
                # robot_env.reset()
                robot_obs, camera_obs = robot_env.get_observation()
                action = controller.forward(robot_obs)
                # robot_env.step(action, blocking=True)
                # move to sigularity position
                joint_angle = [-5.0,-50,-35,68,-9.3,-1.4,7.0]
                joint_radian = [i*3.1415926/180 for i in joint_angle]
                sigularity_pos = {'flexiv_1': {'joint_position': joint_radian, 'gripper_position': [0.0]}}
                robot_env.step(sigularity_pos, blocking=True)

                robot_env.step(sigularity_pos, blocking=True)
                move_leader_pos = True

            robot_obs, camera_obs = robot_env.get_observation()
            # save robot obs
            robot_obs_list.append(robot_obs['flexiv_1']['joint_position'])
            # action = controller.forward(robot_obs)
            # print(action)
            # action = action_linear_interpolation(robot_obs, action)
            # robot_env.step(action)
            if len(robot_obs_list) < 250:
                sigularity_pos = {'flexiv_1': {
                    'joint_position': [-0.25761085748672485, -0.8955100774765015, -0.26642245054244995, 0.5871192216873169,
                                       -0.4577038586139679, -0.5003957748413086, 0.012323334813117981],
                    'gripper_position': [0.0]}}
            else:
                sigularity_pos = {'flexiv_1': {
                    'joint_position': [-0.25761085748672485, -0.6955100774765015, -0.26642245054244995,
                                       0.6871192216873169,
                                       -0.5577038586139679, -0.9003957748413086, 0.032323334813117981],
                    'gripper_position': [0.0]}}
            robot_env.step(sigularity_pos, blocking=False)

            print('Step: ', len(robot_obs_list))
            if len(robot_obs_list) == num_steps:
                break
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            pid = subprocess.run(["pgrep", "-f", "test_controller"], capture_output=True)
            subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
            sys.exit(0)
    # save robot state
    np.save('ours_robot_obs_list.npy', robot_obs_list)




