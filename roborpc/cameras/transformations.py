import math
import time
from math import pi, cos, sin, sqrt, atan2
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


def norm2(a, b, c=0.0):
    return sqrt(a ** 2 + b ** 2 + c ** 2)


def ur_axis_angle_to_quat(axis_angle):
    # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Unit_quaternions
    angle = norm2(*axis_angle)
    axis_normed = [axis_angle[0] / angle, axis_angle[1] / angle, axis_angle[2] / angle]
    s = sin(angle / 2)
    return [s * axis_normed[0], s * axis_normed[1], s * axis_normed[2], cos(angle / 2)]  # xyzw


def quat_to_ur_axis_angle(quaternion):
    # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Unit_quaternions
    # quaternion must be [xyzw]
    angle = 2 * atan2(norm2(quaternion[0], quaternion[1], quaternion[2]), quaternion[3])
    if abs(angle) > 1e-6:
        axis_normed = [quaternion[0] / sin(angle / 2), quaternion[1] / sin(angle / 2),
                       quaternion[2] / sin(angle / 2)]
    else:
        axis_normed = 0.0
    return [axis_normed[0] * angle, axis_normed[1] * angle, axis_normed[2] * angle]



def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    E.g.:
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    Args:
        angle (float): Magnitude of rotation
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (None or np.array): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous matrix that includes the desired rotation
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float32, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

