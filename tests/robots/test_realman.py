import numpy as np

from roborpc.motion_planning.curobo_planner import CuroboPlanner
from roborpc.robots.multi_robots import MultiRobots

if __name__ == '__main__':
    planner = CuroboPlanner()
    multi_realman = MultiRobots()
    multi_realman.connect_now()
    cu_js = multi_realman.get_joint_positions()
    print("Current joint positions:", cu_js)
    goal_js = {"realman_1": [-np.pi, 0.0, 0.0, -np.pi/2, 0.0, -np.pi/2, 0.0]}
    result = planner.plan_js(cu_js, goal_js, use_full_solved_joints_state=True)
    multi_realman.set_robot_state(result, blocking={"realman_1": True})
