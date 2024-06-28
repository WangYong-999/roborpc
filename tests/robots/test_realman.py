import numpy as np

from roborpc.motion_planning.curobo_planner import CuroboPlanner
from roborpc.robots.multi_robots import MultiRobots

if __name__ == '__main__':
    import zerorpc
    planner = CuroboPlanner()
    multi_realman = MultiRobots()
    multi_realman.connect_now()
    print(multi_realman.get_robot_ids())
    cu_js = multi_realman.get_joint_positions()
    goal_js = {"realman_1": [0.0, 0.0, 0.0, -np.pi/2, 0.0, -np.pi/2, 0.0]}
    result = planner.plan_js(cu_js, goal_js, use_full_solved_joints_state=True)
    multi_realman.set_robot_state(result, blocking={"realman_1": True})
