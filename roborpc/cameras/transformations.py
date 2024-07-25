import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import base64
import cv2


def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_quat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_quat()


def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rmat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat


def quat_to_rmat(quat, degrees=False):
    return R.from_quat(quat, degrees=degrees).as_matrix()


### Subtractions ###
def quat_diff(target, source):
    result = R.from_quat(target) * R.from_quat(source).inv()
    return result.as_quat()


def angle_diff(target, source, degrees=False):
    target_rot = R.from_euler("xyz", target, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")


def pose_diff(target, source, degrees=False):
    lin_diff = np.array(target[:3]) - np.array(source[:3])
    rot_diff = angle_diff(target[3:6], source[3:6], degrees=degrees)
    result = np.concatenate([lin_diff, rot_diff])
    return result


### Additions ###
def add_quats(delta, source):
    result = R.from_quat(delta) * R.from_quat(source)
    return result.as_quat()


def add_angles(delta, source, degrees=False):
    delta_rot = R.from_euler("xyz", delta, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)


def add_poses(delta, source, degrees=False):
    lin_sum = np.array(delta[:3]) + np.array(source[:3])
    rot_sum = add_angles(delta[3:6], source[3:6], degrees=degrees)
    result = np.concatenate([lin_sum, rot_sum])
    return result


### MISC ###
def change_pose_frame(pose, frame, degrees=False):
    R_frame = euler_to_rmat(frame[3:6], degrees=degrees)
    R_pose = euler_to_rmat(pose[3:6], degrees=degrees)
    t_frame, t_pose = frame[:3], pose[:3]
    euler_new = rmat_to_euler(R_frame @ R_pose, degrees=degrees)
    t_new = R_frame @ t_pose + t_frame
    result = np.concatenate([t_new, euler_new])
    return result


def rgb_to_base64(rgb, size=None, quality=100):
    height, width = rgb.shape[0], rgb.shape[1]
    if size is not None:
        new_height, new_width = size, int(size * float(width) / height)
        rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # webp seems to be better than png and jpg as a codec, in both compression and quality
    # encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    fmt = ".png"

    # _, rgb_data = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_param)
    _, rgb_data = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(rgb_data).decode("utf-8")


def depth_to_base64(depth, size=None, quality=100):
    height, width = depth.shape[0], depth.shape[1]
    if size is not None:
        new_height, new_width = size, int(size * float(width) / height)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    depth_img = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_img = 255 - depth_img

    # webp seems to be better than png and jpg as a codec, in both compression and quality
    # encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    fmt = ".png"

    _, depth_img_data = cv2.imencode(fmt, depth_img)
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


def apply_transfer(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    # xyz can be 3dim or 4dim (homogeneous) or can be a rotation matrix
    if len(xyz) == 3:
        xyz = np.append(xyz, 1)
    return np.matmul(mat, xyz)[:3]
