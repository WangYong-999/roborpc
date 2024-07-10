from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
import omni.isaac.core.tasks as tasks
from typing import Optional
import numpy as np


class PickPlace(tasks.PickPlace):
    def __init__(
        self,
        name: str,
        gripper_base_link_prim_path: str,
        gripper_joint_prim_names: [list],
        arm_base_link_prim_path: str,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=np.array([0.0515, 0.0515, 0.0515]),
            offset=offset,
        )
        self.my_name = name
        self.gripper_base_link_prim_path = gripper_base_link_prim_path
        self.gripper_joint_prim_names = gripper_joint_prim_names
        self.arm_base_link_prim_path = arm_base_link_prim_path
        return

    def set_robot(self) -> SingleManipulator:
        gripper = ParallelGripper(
            end_effector_prim_path=self.gripper_base_link_prim_path,
            joint_prim_names=self.gripper_joint_prim_names,
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([14.0, 14.0]),
            action_deltas=np.array([-0.45, -0.45]))
        manipulator = SingleManipulator(prim_path=self.arm_base_link_prim_path,
                                        name=self.my_name,
                                        end_effector_prim_name=self.gripper_base_link_prim_path.split("/")[-1],
                                        gripper=gripper)
        # manipulator.set_joints_default_state(positions=joints_default_positions)
        return manipulator

