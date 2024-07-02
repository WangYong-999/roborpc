import base64
import time
from copy import deepcopy

import cv2
import numpy as np

from roborpc.cameras.calibration_utils import HandCameraCalibrator, ThirdPersonCameraCalibrator, calibration_traj, \
    update_calibration_info
from roborpc.common.logger_loader import logger
from roborpc.cameras.transformations import change_pose_frame


def calibrate_camera(
        env,
        camera_id: str,
        controller_id: str,
        robot_id: str,
        is_hand_camera: bool = False,
        step_size: float = 0.01,
        pause_time: float = 0.5,
        image_freq: int = 10,
        reset_robot: bool = True,
):
    cameras = env.cameras
    controllers = env.controllers
    robots = env.robots
    assert pause_time > (env.env_update_rate / 1000)

    intrinsics_dict = cameras.get_camera_intrinsics()[camera_id]
    if is_hand_camera:
        calibrator = HandCameraCalibrator(intrinsics_dict)
    else:
        calibrator = ThirdPersonCameraCalibrator(intrinsics_dict)

    if reset_robot:
        env.reset()

    while True:
        controller_info = controllers.get_info()[controller_id]
        start_time = time.time()

        observation = env.get_observation()
        cameras_obs = observation["cameras"][camera_id]
        robots_obs = observation["robots"][robot_id]

        augmented_images = {camera_id: calibrator.augment_image(camera_id, cameras_obs["color"])}
        action = controllers.forward({"robot_state": robots_obs})
        action[-1] = 0

        comp_time = time.time() - start_time
        sleep_left = (1 / env.control_hz) - comp_time
        if sleep_left > 0:
            time.sleep(sleep_left)

        skip_step = not controller_info["movement_enabled"]
        if not skip_step:
            env.step(action)

        start_calibration = controller_info["success"]
        end_calibration = controller_info["failure"]

        if start_calibration:
            break
        if end_calibration:
            return False

    time.time()
    pose_origin = robots_obs["cartesian_position"]
    i = 0

    while True:
        # Check For Termination #
        controller_info = controllers.get_info()[controller_id]
        if controller_info["failure"]:
            return False

        start_time = time.time()
        take_picture = (i % image_freq) == 0

        if take_picture:
            time.sleep(pause_time)
        observation = env.get_observation()
        cameras_obs = observation["cameras"][camera_id]
        robots_obs = observation["robots"][robot_id]

        if take_picture:
            rgb_img = deepcopy(cameras_obs["color"])
            end_pose = deepcopy(robots_obs["cartesian_position"])
            calibrator.add_sample(camera_id, rgb_img, end_pose)
        augmented_images = {camera_id: calibrator.augment_image(camera_id, cameras_obs["color"])}

        calib_pose = calibration_traj(i * step_size, hand_camera=is_hand_camera)
        desired_pose = change_pose_frame(calib_pose, pose_origin)
        action = np.concatenate([desired_pose, [0]])
        robots.set_ee_pose(action, action_space="cartesian_position", blocking=False)

        comp_time = time.time() - start_time
        sleep_left = (1 / env.env_update_rate) - comp_time
        if sleep_left > 0:
            time.sleep(sleep_left)

        cycle_complete = (i * step_size) >= (2 * np.pi)
        if cycle_complete:
            break
        i += 1
    success = calibrator.is_calibration_accurate(camera_id)
    if not success:
        logger.error(f"Calibration for {camera_id} failed")
        return False
    transformation = calibrator.calibrate(camera_id)
    update_calibration_info(camera_id, transformation)
    logger.success(f"Calibration for {camera_id} successful")
    return True


def rgb_to_base64(rgb, size=None, quality=100):
    height, width = rgb.shape[0], rgb.shape[1]
    if size is not None:
        new_height, new_width = size, int(size * float(width) / height)
        rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # webp seems to be better than png and jpg as a codec, in both compression and quality
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    fmt = ".webp"

    _, rgb_data = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_param)
    return base64.b64encode(rgb_data).decode("utf-8")


def depth_to_base64(depth, size=None, quality=100):
    height, width = depth.shape[0], depth.shape[1]
    if size is not None:
        new_height, new_width = size, int(size * float(width) / height)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    depth_img = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_img = 255 - depth_img

    # webp seems to be better than png and jpg as a codec, in both compression and quality
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    fmt = ".webp"

    _, depth_img_data = cv2.imencode(fmt, depth_img, encode_param)
    return base64.b64encode(depth_img_data).decode("utf-8")


def base64_rgb(isaac_sim_rgb):
    rgb_bytes = base64.b64decode(isaac_sim_rgb)
    rgb_np = np.frombuffer(rgb_bytes, dtype=np.uint8)
    rgb = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
    rgb_np = np.array(rgb)
    return rgb_np


def base64_depth(isaac_sim_depth):
    # Convert depth map to meters
    depth_encoded = isaac_sim_depth
    depth_bytes = base64.b64decode(depth_encoded)
    depth_np = np.frombuffer(depth_bytes, dtype=np.uint8)
    depth_decoded = cv2.imdecode(depth_np, cv2.IMREAD_COLOR)
    depth_unscaled = (255 - np.copy(depth_decoded[:, :, 0]))
    #     depth_scaled = depth_unscaled / 255 * (float(depth["depthMax"]) - float(depth["depthMin"]))
    #     depth_imgs.append(depth_scaled + float(depth["depthMin"]))
    # src_depth = np.array(depth_imgs[0])
    # cur_depth = np.array(depth_imgs[1])
    depth_np = np.array(depth_unscaled)
    return depth_np


