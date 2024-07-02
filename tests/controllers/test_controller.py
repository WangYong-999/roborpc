import subprocess

import numpy as np

from roborpc.common.logger_loader import logger
from roborpc.kinematics_solver.trajectory_interpolation import Polynomial5Interpolation
from roborpc.robot_env import RobotEnv
from roborpc.motion_planning.curobo_planner import CuroboPlanner

if __name__ == '__main__':
    try:
        robot_env = RobotEnv()
        controller = robot_env.controllers
        planner = CuroboPlanner()
        i = 0
        while True:
            try:
                obs = robot_env.get_observation()
                action = controller.forward(obs)
                if i == 0:
                    cu_js = robot_env.robots.get_joint_positions()
                    goal_js = {"realman_1": action['realman_1']["joint_position"]}
                    result = planner.plan_js(cu_js, goal_js, use_full_solved_joints_state=True)
                    logger.info(result)
                    robot_env.robots.set_robot_state(result, blocking={"realman_1": True})
                i += 1
                # interpolate action
                current_joint_postion = np.asarray(obs["realman_1"]["joint_position"][:7])
                goal_joint_position = np.asarray(action['realman_1']["joint_position"][:7])
                # Linear interpolation.
                n_steps = np.ceil(np.abs(np.rad2deg(goal_joint_position) - np.rad2deg(current_joint_postion)))
                n_step = int(np.max(n_steps))
                interpolated_action = np.linspace(current_joint_postion, goal_joint_position, n_step + 1)[1:]
                # gripper_action = np.ones((n_step, 1)) * action['realman_1']["joint_position"][7]
                action = {"realman_1": {"joint_position": interpolated_action.tolist(), "gripper_position": [action['realman_1']["joint_position"][7]]}}
                robot_env.robots.set_robot_state(action, blocking={"realman_1": False})
            except KeyboardInterrupt as e:
                logger.error(e)
                pid = subprocess.run(["pgrep", "-f", "test_controller"], capture_output=True)
                subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                break
    except Exception as e:
        logger.error(e)
        pid = subprocess.run(["pgrep", "-f", "test_controller"], capture_output=True)
        subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
