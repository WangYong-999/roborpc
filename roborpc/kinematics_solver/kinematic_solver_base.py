from abc import ABC


class KinematicSolverBase(ABC):

    @staticmethod
    def fk(q, params):
        pass

    @staticmethod
    def ik(x, params):
        pass
