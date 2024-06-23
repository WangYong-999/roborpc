from abc import ABC


class PlannerBase(ABC):

    @staticmethod
    def plan(start, goal, obstacles):
        pass

    @staticmethod
    def plan_without_obstacles(start, goal, obstacles):
        pass

    @staticmethod
    def plan_with_obstacles(start, goal, obstacles):
        pass

    @staticmethod
    def solve_fk(start, goal, obstacles):
        pass

    @staticmethod
    def solve_ik(start, goal):
        pass


