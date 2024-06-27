from roborpc.robots.multi_robots import MultiRobots

if __name__ == '__main__':
    import zerorpc
    multi_realman = MultiRobots()
    multi_realman.connect_now()
    print(multi_realman.get_robot_ids())
    print(multi_realman.get_robot_state())

    multi_realman.set_joints({"realman_1": [0.0, 0.0, -3.1415926/2, -3.1415926/2,
                              0.0, 0.0, 0.0]},
                             action_space={"realman_1": "joint_position"}, blocking={"realman_1": True})
    # multi_realman.set_gripper(
    #     {"realman_1": [0.1]}, action_space={"realman_1": "gripper_position"}, blocking={"realman_1": True})

    # multi_robots = MultiRobots()
    # s = zerorpc.Server(multi_robots)
    # rpc_port = multi_robots.robot_config['sever_rpc_ports'][0]
    # print(f"RPC Port: {rpc_port}")
    # logger.info(f"RPC Server Start on {rpc_port}")
    # s.bind(f"tcp://0.0.0.0:{rpc_port}")
    # s.run()