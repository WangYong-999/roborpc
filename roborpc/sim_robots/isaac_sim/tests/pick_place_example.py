import os
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "open_usd": os.getcwd() + "/ur5e_with_gripper.usd"})

from omni.isaac.core import World
import numpy as np
from tasks.pick_place import PickPlace
from controllers.pick_place import PickPlaceController
from omni.isaac.core.utils.types import ArticulationAction

my_world = World(stage_units_in_meters=1.0)


target_position = np.array([-0.8, 0.8, 0])
target_position[2] = 0.0915 / 2.0
my_task = PickPlace(name="ur5e_pick_place", target_position=target_position)
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("ur5e_pick_place").get_params()
ur5e_name = task_params["robot_name"]["value"]
my_ur5e = my_world.scene.get_object(ur5e_name)
#initialize the controller
my_controller = PickPlaceController(name="controller", robot_articulation=my_ur5e, gripper=my_ur5e.gripper)
articulation_controller = my_ur5e.get_articulation_controller()
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            print("world resetting")
            my_world.reset()
            my_controller.reset()
        observations = my_world.get_observations()
        #forward the observation values to the controller to get the actions
        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            # This offset needs tuning as well
            end_effector_offset=np.array([0, 0, 0.2]),
        )
        if my_controller.is_done():
            print("done picking and placing")
        # np.save("actions.npy", actions)
        print(actions.joint_positions)
        articulation_controller.apply_action(actions)
simulation_app.close()
