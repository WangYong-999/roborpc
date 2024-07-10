import os
from pathlib import Path

import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation


class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation,
                 robot_description_path: str,
                 rmpflow_config_path: str,
                 urdf_path: str,
                 end_effector_frame_name: str,
                 maximum_substep_size: float = 0.00334,
                 physics_dt: float = 1.0 / 60.0) -> None:
        current_path = Path(__file__).parent.parent.absolute()
        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(current_path, "rmpflow", robot_description_path),
            rmpflow_config_path=os.path.join(current_path, "rmpflow", rmpflow_config_path),
            urdf_path=os.path.join(current_path, "urdf", urdf_path),
            end_effector_frame_name=end_effector_frame_name,
            maximum_substep_size=maximum_substep_size)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)
        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return


def reset(self):
    mg.MotionPolicyController.reset(self)
    self._motion_policy.set_robot_base_pose(
        robot_position=self._default_position, robot_orientation=self._default_orientation
    )
