from roborpc.kinematics_solver.kinematic_solver_base import KinematicBase


class CuroboSolverKinematic(KinematicSolverBase):

    def __init__(self):
        super().__init__()

    def fk(self, joint_angles):
        pass

    def ik(self, pose):
        pass