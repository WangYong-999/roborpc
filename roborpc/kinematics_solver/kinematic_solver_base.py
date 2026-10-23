from abc import ABC


class KinematicSolverBase(ABC):

    @staticmethod
    def forward_kinematics(q, params):
        pass

    @staticmethod
    def inverse_kinematics(x, params):
        pass
