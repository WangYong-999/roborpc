import cv2
import numpy as np
import torch
from torchvision import transforms as T
import droid.utils.data_utils.tensor_utils as TensorUtils
import droid.utils.data_utils.torch_utils as TorchUtils

def converter_helper(data, batchify=True):
    if torch.is_tensor(data):
        pass
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    else:
        raise ValueError

    if batchify:
        data = data.unsqueeze(0)
    return data


def np_dict_to_torch_dict(np_dict, batchify=True):
    torch_dict = {}

    for key in np_dict:
        curr_data = np_dict[key]
        if isinstance(curr_data, dict):
            torch_dict[key] = np_dict_to_torch_dict(curr_data)
        elif isinstance(curr_data, np.ndarray) or torch.is_tensor(curr_data):
            torch_dict[key] = converter_helper(curr_data, batchify=batchify)
        elif isinstance(curr_data, list):
            torch_dict[key] = [converter_helper(d, batchify=batchify) for d in curr_data]
        else:
            raise ValueError

    return torch_dict


def convert_raw_extrinsics_to_Twc(raw_data):
    """
    helper function that convert raw extrinsics (6d pose) to transformation matrix (Twc)
    """
    raw_data = torch.from_numpy(np.array(raw_data))
    pos = raw_data[0:3]
    rot_mat = TorchUtils.euler_angles_to_matrix(raw_data[3:6], convention="XYZ")
    extrinsics = np.zeros((4, 4))
    extrinsics[:3, :3] = TensorUtils.to_numpy(rot_mat)
    extrinsics[:3, 3] = TensorUtils.to_numpy(pos)
    extrinsics[3, 3] = 1.0
    extrinsics = np.linalg.inv(extrinsics)
    return extrinsics


class ImageTransformer:
    def __init__(
        self, remove_alpha=False, bgr_to_rgb=False, augment=False, to_tensor=False, image_path="observation/camera/image"
    ):
        self.image_path = image_path.split("/")
        self.apply_transforms = any([remove_alpha, bgr_to_rgb, augment, to_tensor])

        # Build Composed Transform #
        transforms = []

        if remove_alpha:
            new_transform = T.Lambda(lambda data: data[:, :, :3])
            transforms.append(new_transform)

        if bgr_to_rgb:

            def helper(data):
                if data.shape[-1] == 4:
                    return cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
                return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

            new_transform = T.Lambda(lambda data: helper(data))
            transforms.append(new_transform)

        if augment:
            transforms.append(T.ToPILImage())
            transforms.append(T.AugMix())

        if to_tensor:
            transforms.append(T.ToTensor())

        self.composed_transforms = T.Compose(transforms)

    def forward(self, timestep):
        # Skip If Unnecesary #
        if not self.apply_transforms:
            return timestep

        # Isolate Image Data #
        obs = timestep
        for key in self.image_path:
            obs = obs[key]

        # Apply Transforms #
        for cam_type in obs:
            for i in range(len(obs[cam_type])):
                data = self.composed_transforms(obs[cam_type][i])
                obs[cam_type][i] = data
