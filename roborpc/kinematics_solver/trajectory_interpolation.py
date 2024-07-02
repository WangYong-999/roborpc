from typing import Union, cast, Dict, List

import numpy as np
import numpy.typing as npt


class Polynomial5Interpolation:
    """Point to Point polynomial 5 trajectory interpolation."""

    def __init__(self, start_p: Union[float, npt.NDArray[np.float64]], end_p: Union[float, npt.NDArray[np.float64]],
                 start_v: Union[float, npt.NDArray[np.float64]], end_v: Union[float, npt.NDArray[np.float64]],
                 start_a: Union[float, npt.NDArray[np.float64]], end_a: Union[float, npt.NDArray[np.float64]],
                 start_t: Union[float, npt.NDArray[np.float64]], end_t: Union[float, npt.NDArray[np.float64]]) -> None:
        """Initialize the trajectory.

        Args:
            start_p: Start position.
            end_p: End position.
            start_v: Start velocity.
            end_v: End velocity.
            start_a: Start acceleration.
            end_a: End acceleration.
            start_t: Trajectory start time.
            end_t: Trajectory end time.
        """
        self.start_p = start_p
        self.end_p = end_p
        self.start_v = start_v
        self.end_v = end_v
        self.start_a = start_a
        self.end_a = end_a
        self.start_t = start_t
        self.end_t = end_t

        dist_t = end_t - start_t
        dist_p = end_p - start_p
        dist_a = end_a - start_a

        p1 = dist_t
        p2 = dist_t**2
        p3 = dist_t**3
        p4 = dist_t**4
        p5 = dist_t**5

        self.a0 = start_p
        self.a1 = start_v
        self.a2 = 0.5 * start_a
        self.a3 = (20. * dist_p - (8. * end_v + 12. * start_v) * p1 - (3. * start_a - end_a) * p2) / (2. * p3)
        self.a4 = (-30. * dist_p + (14. * end_v + 16. * start_v) * p1 + (3. * start_a - 2. * end_a) * p2) / (2. * p4)
        self.a5 = (12. * dist_p - 6. * (end_v + start_v) * p1 + dist_a * p2) / (2. * p5)

    def get_point(self, current_time: Union[float, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """Get the reference point according to the current time.

        Args:
            current_time: Current servo time.

        Returns:
            The point position, velocity, and acceleration information at the given time stamp.
        """
        t = current_time - self.start_t
        t2 = t**2
        t3 = t**3
        t4 = t**4
        t5 = t**5

        pos_cmd = self.a0 + self.a1 * t + self.a2 * t2 + self.a3 * t3 + self.a4 * t4 + self.a5 * t5
        vel_cmd = self.a1 + 2. * self.a2 * t + 3. * self.a3 * t2 + 4. * self.a4 * t3 + 5. * self.a5 * t4
        acc_cmd = 2. * self.a2 + 6. * self.a3 * t + 12. * self.a4 * t2 + 20. * self.a5 * t3

        return np.asarray([pos_cmd, vel_cmd, acc_cmd])


class Bezier:
    """Bezier curve generater.

    Ref. https://github.com/torresjrjr/Bezier.py.
    """

    @staticmethod
    def twoPoints(t: Union[int, float], P1: npt.NDArray[np.float64],
                  P2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns a point between P1 and P2, parametised by t.

        Args:
            t: Parameter.
            P1: Point.
            P2: Point.

        Returns:
            Point.
        """
        Q1 = (1 - t) * P1 + t * P2
        return Q1

    @staticmethod
    def points(t: Union[int, float], points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns a list of points interpolated by the Bezier process.

        Args:
            t: Parameter.
            points: Points.

        Returns:
            A list of new points.
        """
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [Bezier.twoPoints(t, points[i1], points[i1 + 1])]
        return np.asarray(newpoints)

    @staticmethod
    def point(t: Union[int, float], points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns a point interpolated by the Bezier process.

        Args:
            t: Parameter.
            points: Points.

        Returns:
            A new point.
        """
        new_points = points
        while len(new_points) > 1:
            new_points = Bezier.points(t, new_points)
        # cast() because new_points.ndim >= 2
        return cast(npt.NDArray[np.float64], new_points[0])

    @staticmethod
    def curve(t_values: npt.NDArray[np.float64], points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns a point interpolated by the Bezier process.

        Args:
            t_values: List of parameters.
            points: List of points. The length of the list must >= 2, i.e.,
                includs at least start and end points, which will degrade to
                the straight line.

        Returns:
            List of curve points.
        """
        if not isinstance(t_values[0], (int, float)):
            raise TypeError('t_values must be an iterable of integers or floats.')

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            curve = np.append(curve, [Bezier.point(t, points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve


def action_linear_interpolation(start_action: Dict[str, Dict[str, List[float]]],
                         end_action: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    """Linear interpolation between two points.

    Args:
        start_action: Start position.
        end_action: End position.


    Returns:
        A list of interpolated points.
    """
    interpolated_action = {}
    for robot_name, action_space_and_values in start_action.items():
        interpolated_action[robot_name] = {}
        for action_space_name, values in action_space_and_values.items():
            if action_space_name == 'joint_position':
                start_value = values
                end_value = end_action[robot_name][action_space_name]
                n_steps = np.ceil(np.abs(np.rad2deg(start_value) - np.rad2deg(end_value)))
                n_step = int(np.max(n_steps))
                interpolated_action[robot_name].update({action_space_name: np.linspace(start_value, end_value, n_step + 1)[1:].tolist()})
            elif action_space_name == 'gripper_position':
                interpolated_action[robot_name].update({action_space_name: end_action[robot_name][action_space_name]})
    return interpolated_action

