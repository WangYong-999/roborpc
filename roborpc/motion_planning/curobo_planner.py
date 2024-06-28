import math
import os
import time
import numpy as np
from typing import Optional, Dict, List
import transforms3d as t3d

import curobo.geom.types
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState as CuroboJointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import (MotionGen, MotionGenConfig,
                                            MotionGenPlanConfig)

from roborpc.common.logger_loader import logger
from roborpc.common.config_loader import config
from roborpc.motion_planning.planner_base import PlannerBase

setup_curobo_logger('warn')
tensor_args = TensorDeviceType()


class CuroboPlanner(PlannerBase):

    def __init__(self) -> None:
        self.graph_mode = False
        self.robot_cfg = None
        self.world_cfg = None
        self.motion_gen = None
        self.plan_config = None
        self.joints_name = None
        self.mp_config = config['roborpc']['motion_planning']['curobo_planner']
        robot_cfg = self.mp_config['robot_cfg']
        world_cfg = self.mp_config['world_cfg']
        self.robot_path = self.mp_config['robot_description']
        self.robot_names_link_names_pair = None
        self.ee_link_name = None
        self.ee_robot_name = None
        self.link_names = None
        self.robot_names_robot_dof_pair = None
        self.load_config(robot_cfg, world_cfg)

    def load_config(
            self,
            robot_cfg: str,
            world_cfg: str,
            obstacle_cuboids_cache: int = 10,
            obstacle_mesh_cache: int = 20,
    ):
        robot_cfg_path = os.path.join(self.robot_path, 'config', robot_cfg)
        world_cfg_path = os.path.join(self.robot_path, 'config', world_cfg)
        self.robot_cfg = load_yaml(robot_cfg_path)['robot_cfg']
        urdf_path = self.robot_cfg['kinematics']['urdf_path']
        self.robot_names_link_names_pair = self.robot_cfg['kinematics']['robot_names_link_names_pair'][0]
        self.robot_names_robot_dof_pair = self.robot_cfg['kinematics']['robot_names_robot_dof_pair'][0]
        self.ee_link_name = self.robot_cfg['kinematics']['ee_link']
        self.robot_cfg['kinematics'].pop('robot_names_link_names_pair')
        self.robot_cfg['kinematics'].pop('robot_names_robot_dof_pair')
        asset_root_path = 'franka_description'
        self.robot_cfg['kinematics']['urdf_path'] = urdf_path
        self.robot_cfg['kinematics']['asset_root_path'] = asset_root_path
        self.robot_cfg['kinematics']['external_asset_path'] = self.robot_path
        self.robot_cfg['kinematics']['external_robot_configs_path'] = self.robot_path

        world_cfg_table = WorldConfig.from_dict(load_yaml(world_cfg_path))
        world_cfg_table.cuboid[0].pose[2] -= 0.02  # table
        world_cfg_mesh = WorldConfig.from_dict(
            load_yaml(world_cfg_path)).get_mesh_world()
        world_cfg_mesh.mesh[0].name += '_mesh'
        world_cfg_mesh.mesh[0].pose[2] = -10.5
        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid,
                                     mesh=world_cfg_mesh.mesh)
        # self.world_cfg = WorldConfig.from_dict(load_yaml(world_cfg_path)).get_obb_world()
        interpolation_steps: int = 2000
        c_checker = CollisionCheckerType.PRIMITIVE
        c_cache: dict = {"obb": 10}
        if self.graph_mode:
            interpolation_steps = 100
        tsteps: int = 30
        trajopt_seeds: int = 4
        finetune_dt_scale: float = 0.9
        collision_activation_distance: float = 0.02
        finetune_iters: int = 200
        grad_iters: int = 125
        ik_seeds: int = 30
        parallel_finetune: bool = True

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            finetune_trajopt_iters=finetune_iters,
            grad_trajopt_iters=grad_iters,
            trajopt_tsteps=tsteps,
            collision_checker_type=c_checker,
            use_cuda_graph=True,
            collision_cache=c_cache,
            position_threshold=0.005,  # 5 mm
            rotation_threshold=0.05,
            num_ik_seeds=ik_seeds,
            num_graph_seeds=trajopt_seeds,
            num_trajopt_seeds=trajopt_seeds,
            interpolation_dt=0.025,
            interpolation_steps=interpolation_steps,
            collision_activation_distance=collision_activation_distance,
            trajopt_dt=0.25,
            finetune_dt_scale=finetune_dt_scale,
            maximum_trajectory_dt=0.1,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        # self.plan_config = MotionGenPlanConfig(enable_graph=graph_mode,
        #                                        enable_graph_attempt=4,
        #                                        max_attempts=10,
        #                                        enable_finetune_trajopt=True)
        self.plan_config = MotionGenPlanConfig(
            max_attempts=100,
            enable_graph_attempt=1,
            disable_graph_attempt=20,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            enable_graph=self.graph_mode,
            timeout=60,
            enable_opt=not self.graph_mode,
            # need_graph_success=not self.graph_mode,
            parallel_finetune=parallel_finetune,
        )

        self.joints_name = self.robot_cfg['kinematics']['cspace'][
            'joint_names']

        # Warm Up
        logger.info('Curobo Warm Up Necessarily...')
        start_time = time.time()
        self.warm_up()
        logger.info(f'Curobo Warm Up Done! Cost time is {time.time() - start_time}')

    def warm_up(self) -> None:
        self.motion_gen.warmup(enable_graph=self.graph_mode,
                               warmup_js_trajopt=False,
                               parallel_finetune=True)
        for robot_name, link_name in self.robot_names_link_names_pair.items():
            if link_name == self.ee_link_name:
                self.ee_robot_name = robot_name

    def plan(self,
             joints_state: Dict[str, List[float]],
             goal_pose: Dict[str, List[float]],
             use_full_solved_joints_state: bool = False,
             select_joints_num: int = 10) -> Dict[str, Dict[str, List[List[float]]]]:
        joints_name, joints_value = [], []
        for robot_name, joint_value in joints_state.items():
            joints_name.extend(self.joints_name[self.robot_names_robot_dof_pair[robot_name][0]:
                                                self.robot_names_robot_dof_pair[robot_name][1]])
            joints_value.extend(joint_value)
        cu_js = CuroboJointState(
            position=tensor_args.to_device(joints_value),
            joint_names=joints_name,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        arm_ik_goal = CuroboPose(position=tensor_args.to_device(goal_pose[self.ee_robot_name][:3]),
                                 quaternion=tensor_args.to_device(
                                     t3d.euler.euler2quat(*goal_pose[self.ee_robot_name][3:], axes='sxyz')))
        link_poses = {}
        for robot_name, link_name in self.robot_names_link_names_pair.items():
            if link_name != self.ee_link_name:
                link_poses[link_name] = CuroboPose(position=tensor_args.to_device(goal_pose[robot_name][:3]),
                                                   quaternion=tensor_args.to_device(
                                                       t3d.euler.euler2quat(*goal_pose[robot_name][3:], axes='sxyz')))

        start_time = time.time()
        result = self.motion_gen.plan_single(
            start_state=cu_js.unsqueeze(0),
            goal_pose=arm_ik_goal,
            plan_config=self.plan_config,
            link_poses=link_poses,
        )
        cost_time = time.time() - start_time

        success = result.success.item()
        solved_joints_state = {}
        if success:
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(cmd_plan)
            cmd_plan = cmd_plan.get_ordered_joint_state(self.joints_name)
            len_solve_joints_array = len(cmd_plan.position.cpu().numpy()) - 1
            skip_joints_array = math.ceil(len_solve_joints_array / select_joints_num)

            if not use_full_solved_joints_state:
                for robot_name, joint_value in joints_state.items():
                    select = self.robot_names_robot_dof_pair[robot_name]
                    solved_joints_state[robot_name] = {
                        "joint_position": list(np.concatenate([cmd_plan.position.cpu().numpy()[
                                                               :len_solve_joints_array:skip_joints_array,
                                                               select[0]:select[1]],
                                                               cmd_plan.position.cpu().numpy()[len_solve_joints_array,
                                                               select[0]:select[1]]], axis=0)),
                        "joint_velocity": list(np.concatenate([cmd_plan.velocity.cpu().numpy()[
                                                               :len_solve_joints_array:skip_joints_array,
                                                               select[0]:select[1]],
                                                               cmd_plan.velocity.cpu().numpy()[len_solve_joints_array,
                                                               select[0]:select[1]]], axis=0)),
                        "joint_acceleration": list(np.concatenate([cmd_plan.acceleration.cpu().numpy()[
                                                                   :len_solve_joints_array:skip_joints_array,
                                                                   select[0]:select[1]],
                                                                   cmd_plan.acceleration.cpu().numpy()[
                                                                   len_solve_joints_array, select[0]:select[1]]],
                                                                  axis=0)),
                    }

            else:
                for robot_name, joint_value in joints_state.items():
                    select = self.robot_names_robot_dof_pair[robot_name]
                    solved_joints_state[robot_name] = {
                        "joint_position": list(cmd_plan.position.cpu().numpy()[:len_solve_joints_array + 1,
                                               select[0]:select[1]]),
                        "joint_velocity": list(cmd_plan.velocity.cpu().numpy()[:len_solve_joints_array + 1,
                                               select[0]:select[1]]),
                        "joint_acceleration": list(cmd_plan.acceleration.cpu().numpy()[:len_solve_joints_array + 1,
                                                   select[0]:select[1]])
                    }

            final_joints_len = (len_solve_joints_array + 1) if use_full_solved_joints_state else select_joints_num
            logger.success(f"Plan Solve Success. Cost time is {cost_time}.\n"
                           f"Use full solved joints state is {use_full_solved_joints_state}, "
                           f"Solved joints number is  {final_joints_len}.\n"
                           f"Solved full joints {self.motion_gen.kinematics.joint_names} ")
            return solved_joints_state
        else:
            logger.warning(f"Plan did not converge to a solution. "
                           f"Because {'the start/end state is collision' if result.valid_query else ''} or "
                           f"the goal state is {result.status}.")
            return solved_joints_state

    def plan_js(self,
                joints_state: Dict[str, List[float]],
                goal_joints_state: Dict[str, List[float]],
                use_full_solved_joints_state: bool = False,
                select_joints_num: int = 10) -> Dict[str, Dict[str, List[List[float]]]]:
        joints_name, joints_value = [], []
        for robot_name, joint_value in joints_state.items():
            joints_name.extend(self.joints_name[self.robot_names_robot_dof_pair[robot_name][0]:
                                                self.robot_names_robot_dof_pair[robot_name][1]])
            joints_value.extend(joint_value)
        cu_js = CuroboJointState(
            position=tensor_args.to_device(joints_value),
            velocity=tensor_args.to_device([0.0] * len(joints_value)),
            acceleration=tensor_args.to_device([0.0] * len(joints_value)),
            joint_names=joints_name,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        joints_name, joints_value = [], []
        for robot_name, joint_value in goal_joints_state.items():
            joints_name.extend(self.joints_name[self.robot_names_robot_dof_pair[robot_name][0]:
                                                self.robot_names_robot_dof_pair[robot_name][1]])
            joints_value.extend(joint_value)
        goal_js = CuroboJointState(
            position=tensor_args.to_device(joints_value),
            velocity=tensor_args.to_device([0.0] * len(joints_value)),
            acceleration=tensor_args.to_device([0.0] * len(joints_value)),
            joint_names=joints_name,
        )
        goal_js = goal_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        print(goal_js)
        print(cu_js)
        start_time = time.time()
        result = self.motion_gen.plan_single_js(
            start_state=cu_js.unsqueeze(0),
            goal_state=goal_js.unsqueeze(0),
            plan_config=self.plan_config,
        )
        cost_time = time.time() - start_time

        success = result.success.item()
        solved_joints_state = {}
        if success:
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(cmd_plan)
            cmd_plan = cmd_plan.get_ordered_joint_state(self.joints_name)
            len_solve_joints_array = len(cmd_plan.position.cpu().numpy()) - 1
            skip_joints_array = math.ceil(len_solve_joints_array / select_joints_num)

            if not use_full_solved_joints_state:
                for robot_name, joint_value in joints_state.items():
                    select = self.robot_names_robot_dof_pair[robot_name]
                    solved_joints_state[robot_name] = {
                        "joint_position": np.concatenate([cmd_plan.position.cpu().numpy()[
                                                               :len_solve_joints_array:skip_joints_array,
                                                               select[0]:select[1]],
                                                               [cmd_plan.position.cpu().numpy()[len_solve_joints_array,
                                                               select[0]:select[1]]]], axis=0).tolist(),
                        "joint_velocity": np.concatenate([cmd_plan.velocity.cpu().numpy()[
                                                               :len_solve_joints_array:skip_joints_array,
                                                               select[0]:select[1]],
                                                               [cmd_plan.velocity.cpu().numpy()[len_solve_joints_array,
                                                               select[0]:select[1]]]], axis=0).tolist(),
                        "joint_acceleration": np.concatenate([cmd_plan.acceleration.cpu().numpy()[
                                                                   :len_solve_joints_array:skip_joints_array,
                                                                   select[0]:select[1]],
                                                                   [cmd_plan.acceleration.cpu().numpy()[
                                                                   len_solve_joints_array, select[0]:select[1]]]],
                                                                  axis=0).tolist(),
                    }

            else:
                for robot_name, joint_value in joints_state.items():
                    select = self.robot_names_robot_dof_pair[robot_name]
                    solved_joints_state[robot_name] = {
                        "joint_position": cmd_plan.position.cpu().numpy()[:len_solve_joints_array + 1,
                                               select[0]:select[1]].tolist(),
                        "joint_velocity": cmd_plan.velocity.cpu().numpy()[:len_solve_joints_array + 1,
                                               select[0]:select[1]].tolist(),
                        "joint_acceleration": cmd_plan.acceleration.cpu().numpy()[:len_solve_joints_array + 1,
                                                   select[0]:select[1]].tolist()
                    }

            final_joints_len = (len_solve_joints_array + 1) if use_full_solved_joints_state else select_joints_num
            logger.success(f"Plan Solve Success. Cost time is {cost_time}.\n"
                           f"Use full solved joints state is {use_full_solved_joints_state}, "
                           f"Solved joints number is  {final_joints_len}.\n"
                           f"Solved full joints {self.motion_gen.kinematics.joint_names} ")
            return solved_joints_state
        else:
            logger.warning(f"Plan did not converge to a solution. "
                           f"Because {'the start/end state is collision' if result.valid_query else ''} or "
                           f"the goal state is {result.status}.")
            return solved_joints_state


if __name__ == '__main__':
    planner = CuroboPlanner()
    joint_angles = {
        'realman_1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

    print(planner.plan_js(joints_state=joint_angles,
                          goal_joints_state={'realman_1': [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]},
                          use_full_solved_joints_state=False))
