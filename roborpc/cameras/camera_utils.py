import base64

import cv2
import numpy as np


def rgb_to_base64(rgb, size=None, quality=10):
    height, width = rgb.shape[0], rgb.shape[1]
    if size is not None:
        new_height, new_width = size, int(size * float(width) / height)
        rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # webp seems to be better than png and jpg as a codec, in both compression and quality
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    fmt = ".webp"

    _, rgb_data = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_param)
    return base64.b64encode(rgb_data).decode("utf-8")


def depth_to_base64(depth, size=None, quality=10):
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
