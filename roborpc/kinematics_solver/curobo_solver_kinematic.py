import os
from typing import Dict, List
import numpy as np
import transforms3d as t3d
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger
from roborpc.kinematics_solver.kinematic_solver_base import KinematicSolverBase

tensor_args = TensorDeviceType()


class CuroboSolverKinematic(KinematicSolverBase):

    def __init__(self):
        super().__init__()
        self.ik_config = config['roborpc']['kinematics_solver']['curobo_solver_kinematic']
        robot_cfg = self.ik_config['robot_cfg']
        world_cfg = self.ik_config['world_cfg']
        self.robot_path = self.ik_config['robot_description']

        self.robot_names_link_names_pair = None
        self.robot_dof_link_names_pair = None
        self.default_joint = None
        self.link_retract_pose = None
        self.link_names = None
        self.ee_link_name = None
        self.robot_cfg = None
        self.world_cfg = None
        self.plan_config = None
        self.joints_name = None
        self.ik_solver = None

        self.load_config(robot_cfg, world_cfg)

    def load_config(
            self,
            robot_cfg: str,
            world_cfg: str,
            collision_check: bool = True,
            obstacle_cuboids_cache: int = 10,
            obstacle_mesh_cache: int = 20,
    ):
        robot_cfg_path = os.path.join(self.robot_path, 'config', robot_cfg)
        world_cfg_path = os.path.join(self.robot_path, 'config', world_cfg)
        self.robot_cfg = load_yaml(robot_cfg_path)['robot_cfg']
        self.ee_link_name = self.robot_cfg['kinematics']['ee_link']
        self.link_names = self.robot_cfg['kinematics']['link_names']
        self.robot_names_link_names_pair = self.robot_cfg['kinematics']['robot_names_link_names_pair'][0]
        self.robot_dof_link_names_pair = self.robot_cfg['kinematics']['robot_dof_link_names_pair'][0]
        self.robot_cfg['kinematics'].pop('robot_names_link_names_pair')
        self.robot_cfg['kinematics'].pop('robot_dof_link_names_pair')
        self.default_joint = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        urdf_path = self.robot_cfg['kinematics']['urdf_path']
        asset_root_path = '../robot_description' + os.path.dirname(urdf_path)
        self.robot_cfg['kinematics']['urdf_path'] = urdf_path
        self.robot_cfg['kinematics']['asset_root_path'] = asset_root_path
        self.robot_cfg['kinematics']['external_asset_path'] = self.robot_path
        self.robot_cfg['kinematics']['external_robot_configs_path'] = self.robot_path

        world_cfg_table = WorldConfig.from_dict(load_yaml(world_cfg_path))
        world_cfg_table.cuboid[0].pose[2] -= 0.02  # table
        world_cfg_mesh = WorldConfig.from_dict(load_yaml(world_cfg_path)).get_mesh_world()
        world_cfg_mesh.mesh[0].name += '_mesh'
        world_cfg_mesh.mesh[0].pose[2] = -10.5
        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg_mesh.mesh)

        self.joints_name = self.robot_cfg['kinematics']['cspace']['joint_names']

        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.001,
            num_seeds=20,
            self_collision_check=collision_check,
            self_collision_opt=collision_check,
            tensor_args=tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                'obb': obstacle_cuboids_cache,
                'mesh': obstacle_mesh_cache
            },
            # use_fixed_samples=True,
        )
        self.ik_solver = IKSolver(ik_config)

        # Warm Up
        logger.info('IK Warm Up...')
        self.warm_up()
        logger.info('IK Warm Up Done!')

    def warm_up(self):
        link_poses = {}
        q_sample = self.ik_solver.sample_configs(1)
        kin_state = self.ik_solver.fk(q_sample)
        ee_pose = CuroboPose(position=tensor_args.to_device(kin_state.ee_position),
                             quaternion=tensor_args.to_device(kin_state.ee_quaternion))
        for link_name in self.link_names:
            if link_name != self.ee_link_name:
                link_poses[link_name] = kin_state.link_pose[link_name].to_list()
        for i in range(5):
            _ = self.ik_solver.solve_batch(ee_pose, link_poses=link_poses)
        torch.cuda.synchronize()

    def forward_kinematics(self, joint_angles: Dict[str, List[float]]) -> Dict[str, List[float]]:
        link_pose = {}
        print(self.robot_names_link_names_pair)
        joint_angles_list = []
        for link_name, robot_name in self.robot_names_link_names_pair.items():
            joint_angles_list.append(joint_angles[robot_name])
        state = self.ik_solver.fk(tensor_args.to_device(joint_angles_list))
        ee_position = state.ee_position.cpu().numpy()[0]
        ee_wxyz = state.ee_quaternion.cpu().numpy()[0]
        link_pose[self.robot_names_link_names_pair[self.ee_link_name]] = np.concatenate(
            [ee_position, t3d.euler.quat2euler(ee_wxyz, axes='sxyz')]).tolist()
        for link_name in self.link_names:
            if link_name != self.ee_link_name:
                link_pose[self.robot_names_link_names_pair[link_name]] = state.link_pose[link_name].to_list()
        # right_position = state.link_pose['ee_link_1'].position.cpu().numpy()[0]
        # right_wxyz = state.link_pose['ee_link_1'].quaternion.cpu().numpy()[0]
        return link_pose

    def inverse_kinematics(self, pose: Dict[str, List[float]]) -> Dict[str, List[float]]:
        ee_link_name = self.robot_names_link_names_pair[self.ee_link_name]
        ee_pose = CuroboPose(position=tensor_args.to_device(pose[ee_link_name][:3]),
                             quaternion=tensor_args.to_device(
                                 t3d.euler.euler2quat(*pose[ee_link_name][3:], axes='sxyz')))
        link_poses = {}
        for link_name, robot_name in self.robot_names_link_names_pair.items():
            if link_name != self.ee_link_name:
                link_poses[link_name] = CuroboPose(position=tensor_args.to_device(pose[robot_name][:3]),
                                                   quaternion=tensor_args.to_device(
                                                       t3d.euler.euler2quat(*pose[robot_name][3:], axes='sxyz')))
        result = self.ik_solver.solve_batch(goal_pose=ee_pose, link_poses=link_poses)
        torch.cuda.synchronize()
        success = torch.any(result.success)
        joint_angles = {}
        if success:
            cmd_plan = result.js_solution
            cmd_plan = cmd_plan.get_ordered_joint_state(self.joints_name)
            result_position = list(cmd_plan.position.cpu().numpy()[0][0])
            for link_name, robot_name in self.robot_names_link_names_pair.items():
                joints_result = []
                for i in range(self.robot_dof_link_names_pair[link_name]):
                    joints_result.append(result_position.pop(0))
                joint_angles[robot_name] = joints_result
            logger.success(f'Ik Solve Success. Solved joints is {self.ik_solver.kinematics.joint_names}.\n'
                           f'Joints angle is {joint_angles}')
            return joint_angles
        else:
            logger.warning('Ik solve did not converge to a solution.')
            return joint_angles


if __name__ == '__main__':
    solver = CuroboSolverKinematic()
    joint_angles = {
        'realman_1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

    pose = solver.forward_kinematics(joint_angles)
    print(pose)
    joint_angles = solver.inverse_kinematics(pose)
    print(joint_angles)
