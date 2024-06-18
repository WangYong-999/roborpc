from __future__ import annotations

import traceback

import requests

command_startup = {
    'command': 'startup_isaac'
}
command_shutdown = {
    'command': 'shutdown_isaac'
}

command_load_task_simulation = {
    'scene_task_id': '12312313',
    'command': 'load_task_simulation',
    'scene_usd': '/nfsroot/scenes/shanghai/laboratory_usd/lab_colmo_cab_can_1_7_auto_open.usd',
    'scene_task_name': 'pick_and_place_into_receptacle',
    'source_object_prim_path': '/root/Can_27',
    'target_object_prim_path': '/Mobile_Robot_Gen_2/Tray_Arm/Tray_Collider',
    'robot_init_pose': [352.0, -79.0, 0.0, 0.0, 0.0, -90.0],
    'task_command': 'pick Can_2 and place into Plate_1',
    'robot_init_arm_joints':[1.5710250036267028,
                                              -1.3962033283540387,
                                              -1.0474870365975733,
                                              2.7923556854543814,
                                              -0.5235441609572392,
                                              -3.302728092494427e-5,
                                              -1.5707987427375252
                                              ],
    'robot_init_gripper_degrees':[0.0, 0.0, -20.0, -75.0, 20.0, 75.0, 20.0, 75.0],
    'robot_init_body_position':[22.0, 0.0],
    'collector_path': '/home/jz08/Log'
}
command_retrieve_object_placement_info = {
    'command': 'retrieve_object_placement_info',
    'type': 'on',
    'source_obj': 'Can_2',
    'target_obj': 'Plate_1'
}
command_retrieve_object_bbox_info = {
    'command': 'retrieve_object_bbox_info',
    'object_name': 'Can_2'
}

command_check_source_on_target_aabb = {'command': 'check_source_on_target_aabb'}
command_check_source_in_target_aabb = {'command': 'check_source_in_target_aabb'}
command_check_pick_object = {'command': 'check_pick_object'}
command_check_object_into_roi = {'command': 'check_object_into_roi'}
command_check_object_near_object = {'command': 'check_object_near_object'}
command_check_object_into_receptacle = {'command': 'check_object_into_receptacle'}


post_response1 = requests.post('http://127.0.0.1:6026/', json=command_startup)
print(post_response1.status_code)
print(post_response1.json())

post_response2 = requests.post('http://127.0.0.1:6028/', json=command_load_task_simulation)
print(post_response2.json())
