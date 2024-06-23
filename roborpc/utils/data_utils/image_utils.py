import cv2
import base64
from PIL import Image
import open3d as o3d


class RGBDepth:
    """Class for the current RGB, depth and point cloud fetched from the robot.
    Args:
        rgb (np.array): RGB image fetched from the robot
        depth (np.array): depth map fetched from the robot
        pts (np.array [(x,y,z)]): array of x,y,z coordinates of the pointcloud corresponding
        to the rgb and depth maps.
    """

    rgb: np.array
    depth: np.array
    ptcloud: np.array

    def __init__(self, rgb, depth, pts):
        self.rgb = rgb
        self.depth = depth
        self.ptcloud = pts.reshape(rgb.shape)

    def get_pillow_image(self):

        return Image.fromarray(self.rgb, "RGB")

    def get_bounds_for_mask(self, mask):
        """for all points in the mask, returns the bounds as an axis-aligned bounding box.
        """
        if mask is None:
            return None
        points = self.ptcloud[np.where(mask == True)]
        points = xyz_pyrobot_to_canonical_coords(points)
        points = o3d.utility.Vector3dVector(points)
        obb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
        return np.concatenate([obb.get_min_bound(), obb.get_max_bound()])

    def get_coords_for_point(self, point):
        """fetches xyz from the point cloud in pyrobot coordinates and converts it to
        canonical world coordinates.
        """
        if point is None:
            return None
        xyz_p = self.ptcloud[point[1], point[0]]
        return xyz_pyrobot_to_canonical_coords(xyz_p)

    def to_struct(self, size=None, quality=10):
        rgb = self.rgb
        depth = self.depth
        height, width = rgb.shape[0], rgb.shape[1]
        if size is not None:
            new_height, new_width = size, int(size * float(width) / height)
            rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        depth_img = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_img = 255 - depth_img

        # webp seems to be better than png and jpg as a codec, in both compression and quality
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        fmt = ".webp"

        _, rgb_data = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_param)
        _, depth_img_data = cv2.imencode(fmt, depth_img, encode_param)
        return {
            "rgb": base64.b64encode(rgb_data).decode("utf-8"),
            "depth_img": base64.b64encode(depth_img_data).decode("utf-8"),
            "depth_max": str(np.max(depth)),
            "depth_min": str(np.min(depth)),
        }

    def b64_to_np_array(self, data):
        # Decode rgb map
        rgb_bytes = base64.b64decode(data["prevRgbImg"])
        rgb_np = np.frombuffer(rgb_bytes, dtype=np.uint8)
        rgb = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
        src_img = np.array(rgb)
        height, width, _ = src_img.shape

        # Convert depth map to meters
        depth_imgs = []
        for i, depth in enumerate([postData["prevDepth"], postData["depth"]]):
            depth_encoded = depth["depthImg"]
            depth_bytes = base64.b64decode(depth_encoded)
            depth_np = np.frombuffer(depth_bytes, dtype=np.uint8)
            depth_decoded = cv2.imdecode(depth_np, cv2.IMREAD_COLOR)
            depth_unscaled = (255 - np.copy(depth_decoded[:, :, 0]))
            depth_scaled = depth_unscaled / 255 * (float(depth["depthMax"]) - float(depth["depthMin"]))
            depth_imgs.append(depth_scaled + float(depth["depthMin"]))
        src_depth = np.array(depth_imgs[0])
        cur_depth = np.array(depth_imgs[1])
