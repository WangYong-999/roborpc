import os
import time
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import serial
from dataclasses import dataclass
from pynput import keyboard
from pynput.keyboard import Key

from thirty_part.dynamixel.dynamixel import DynamixelRobot
from common.config_loader import config as common_config


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, int, int]
    """The gripper config of GELLO. This is a tuple of (gripper_joint_id, degrees in open_position, degrees in closed_position)."""

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
            self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )


joint_offsets_list = common_config["droid"]["robot"]["gello_offset"]
gripper_offset_list = common_config["droid"]["robot"]["gello_gripper_offset"]
PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
    # xArm
    # "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0": DynamixelRobotConfig(
    #     joint_ids=(1, 2, 3, 4, 5, 6, 7),
    #     joint_offsets=(
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         -1 * np.pi / 2 + 2 * np.pi,
    #         1 * np.pi / 2,
    #         1 * np.pi / 2,
    #     ),
    #     joint_signs=(1, 1, 1, 1, 1, 1, 1),
    #     gripper_config=(8, 279, 279 - 50),
    # ),
    # panda
    common_config["droid"]["robot"]["gello_port"]: DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=tuple(joint_offsets_list),
        joint_signs=(1, -1, 1, 1, 1, -1, 1),
        gripper_config=(8, gripper_offset_list[0], gripper_offset_list[1]),
    ),
    # Left UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            0,
            1 * np.pi / 2 + np.pi,
            np.pi / 2 + 0 * np.pi,
            0 * np.pi + np.pi / 2,
            np.pi - 2 * np.pi / 2,
            -1 * np.pi / 2 + 2 * np.pi,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 20, -22),
    ),
    # Right UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            np.pi + 0 * np.pi,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            1 * np.pi,
            3 * np.pi / 2,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 286, 248),
    ),
}


class GelloPolicy:
    def __init__(
            self,
            port: str,
            dynamixel_config: Optional[DynamixelRobotConfig] = None,
            right_controller: bool = True,
            max_lin_vel: float = 1,
            max_rot_vel: float = 1,
            max_gripper_vel: float = 1,
            spatial_coeff: float = 1,
            pos_action_gain: float = 5,
            rot_action_gain: float = 2,
            gripper_action_gain: float = 3,
            rmat_reorder: list = [-2, -1, -3, 4],
    ):
        self.key_button = False
        start_joints = np.array(common_config["droid"]["robot"]["start_joints"])

        if dynamixel_config is not None:
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints
            )
        else:
            assert os.path.exists(port), port
            assert port in PORT_CONFIG_MAP, f"Port {port} not in config map"

            config = PORT_CONFIG_MAP[port]
            self._robot = config.make_robot(port=port, start_joints=start_joints)
        # button_A_port = common_config["droid"]["robot"]["button_A_port"]
        # button_B_port = common_config["droid"]["robot"]["button_B_port"]
        # self.button_A = serial.Serial(button_A_port, baudrate=115200, timeout=0.02, xonxoff=False, rtscts=False,
        #                               dsrdtr=False)
        # self.button_B = serial.Serial(button_B_port, baudrate=115200, timeout=0.02, xonxoff=False, rtscts=False,
        #                               dsrdtr=False)

        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self.reset_state()

        # Checkout jonits offset
        print(self._robot.get_joint_state())
        print(common_config["droid"]["robot"]["start_joints"])
        assert np.allclose(self._robot.get_joint_state(), common_config["droid"]["robot"]["start_joints"],
                           rtol=2, atol=2)

        self.button_A_pressed = False
        self.button_B_pressed = False
        run_threaded_command(self.run_key_listen)
        # Start State Listening Thread #
        run_threaded_command(self._update_internal_state)
        # wait gello update joints angle
        time.sleep(3)

    def run_key_listen(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        try:
            if key == Key.scroll_lock:
                self.button_A_pressed = True
            if key == Key.pause:
                self.button_B_pressed = True
        except AttributeError:
            ...

    def on_release(self, key):
        try:
            if key == Key.scroll_lock:
                self.button_A_pressed = True
            if key == Key.pause:
                self.button_B_pressed = True
        except AttributeError:
            ...

    def reset_state(self):
        self._state = {
            "gello_joints": {},
            "poses": {},
            "buttons": {"A": False, "B": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self._goto_start_pos = False
        self.move_start_flag = True
        self.move_start_num = 0

    def reset_gello_start_pos(self):
        self._goto_start_pos = True

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            # Read Controller
            time_since_read = time.time() - last_read_time
            dyna_joints = self._robot.get_joint_state()
            # print(f"dyna_joints: {dyna_joints}")
            current_q = dyna_joints[:-1]  # last one dim is the gripper
            current_gripper = dyna_joints[-1]  # last one dim is the gripper
            # Save Info #
            self._state["gello_joints"][self.controller_id] = dyna_joints
            self._state["controller_on"] = time_since_read < num_wait_sec
            # Determine Control Pipeline #
            # poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            # button_A_data = self.button_A.readline()
            # button_B_data = self.button_B.readline()
            # if button_A_data == b'\x00':
            #     self.button_A_pressed = True
            # else:
            #     self.button_A_pressed = False
            #
            # if button_B_data == b'\x00':
            #     self.button_B_pressed = True
            # else:
            #     self.button_B_pressed = False
            self.update_sensor = True
            if self.button_A_pressed:
                print("button_A_pressed")
            if self.button_B_pressed:
                print("button_B_pressed")
            self._state["buttons"] = {"A": self.button_A_pressed, "B": self.button_B_pressed}
            self._state["movement_enabled"] = True
            self._state["controller_on"] = True
            last_read_time = time.time()

            self.button_A_pressed = False
            self.button_B_pressed = False

    def _process_reading(self):
        gello_joints = np.asarray(self._state["gello_joints"][self.controller_id])
        self.gello_joints = {"gello_joints": gello_joints}

    def _go_start_joints(self, state_dict):
        # going to start position
        print("Going to start position")
        # get gello data
        while self._state["gello_joints"] == {}:
            print("gello joints is empty")
        start_pos = np.asarray(self._state["gello_joints"][self.controller_id])

        # get obs data (Franka)
        obs_joints = state_dict["joint_positions"]
        obs_gripper = state_dict["gripper_position"]
        if type(obs_gripper) == list:
            obs_gripper_new = obs_gripper[0]
        else:
            obs_gripper_new = obs_gripper
        obs_pos = np.concatenate([obs_joints, [obs_gripper_new]])

        abs_deltas = np.abs(start_pos - obs_pos)
        id_max_joint_delta = np.argmax(abs_deltas)

        max_joint_delta = 0.8
        if abs_deltas[id_max_joint_delta] > max_joint_delta:
            id_mask = abs_deltas > max_joint_delta
            print()
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                    ids,
                    abs_deltas[id_mask],
                    start_pos[id_mask],
                    obs_pos[id_mask],
            ):
                print(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            return

        print(f"Start pos: {len(start_pos)}", f"Joints: {len(obs_pos)}")
        assert len(start_pos) == len(
            obs_pos
        ), f"agent output dim = {len(start_pos)}, but env dim = {len(obs_pos)}"

        max_delta = 0.05
        self.move_start_num += 1
        if self.move_start_num == 24:
            self.move_start_flag = False
        if self.move_start_flag:
            command_joints = np.asarray(self._state["gello_joints"][self.controller_id])
            current_joints = np.concatenate([state_dict["joint_positions"], [state_dict["gripper_position"]]])
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            return current_joints + delta

        joints = np.concatenate([state_dict["joint_positions"], [state_dict["gripper_position"]]])
        action = np.asarray(self._state["gello_joints"][self.controller_id])
        if (action - joints > 0.5).any():
            print("Action is too big")

            # print which joints are too big
            joint_index = np.where(action - joints > 0.5)
            for j in joint_index:
                print(
                    f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                )
            exit()
        self._goto_start_pos = True

    def _calculate_action(self, state_dict, include_info=False):
        # Read Sensor #
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False

        # gello
        if include_info:
            return np.asarray(self._state["gello_joints"][self.controller_id]), {}
        else:
            return np.asarray(self._state["gello_joints"][self.controller_id])

    def get_info(self):
        return {
            "success": self._state["buttons"]["A"],
            "failure": self._state["buttons"]["B"],
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }

    def forward(self, obs_dict, include_info=False):
        if not self._goto_start_pos:
            self._go_start_joints(obs_dict["robot_state"])
        return self._calculate_action(obs_dict["robot_state"], include_info=include_info)
