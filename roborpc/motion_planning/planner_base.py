from abc import ABC
from typing import Dict, List


class PlannerBase(ABC):

    @staticmethod
    def plan(joints_state: Dict[str, List[float]],
             goal_pose: Dict[str, List[float]],
             use_full_solved_joints_state: bool = False,
             select_joints_num: int = 10) -> Dict[str, Dict[str, List[List[float]]]]:
        """
        Plan a path from the current joints state to the goal pose.

        Args:
            joints_state (Dict[str, List[float]]): Current joints state.
            goal_pose (Dict[str, List[float]]): Goal pose.
            use_full_solved_joints_state (bool, optional): Whether to use the full solved joints state or not. Defaults to False.
            select_joints_num (int, optional): Number of joints to select for the path. Defaults to 10.

        Returns:
            Dict[str, List[List[float]]]: Path as a list of joints states.
        """
        raise NotImplementedError

    @staticmethod
    def plan_js(joints_state: Dict[str, List[float]],
                goal_joints_state: Dict[str, List[float]],
                use_full_solved_joints_state: bool = False,
                select_joints_num: int = 10) -> Dict[str, Dict[str, List[List[float]]]]:
        """
        Plan a path from the current joints state to the goal joints state.

        Args:
            joints_state (Dict[str, List[float]]): Current joints state.
            goal_joints_state (Dict[str, List[float]]): Goal joints state.
            use_full_solved_joints_state (bool, optional): Whether to use the full solved joints state or not. Defaults to False.
            select_joints_num (int, optional): Number of joints to select for the path. Defaults to 10.

        Returns:
            Dict[str, List[List[float]]]: Path as a list of joints states.
        """
