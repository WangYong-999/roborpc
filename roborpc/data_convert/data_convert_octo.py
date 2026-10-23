import sys
from typing import Iterator, Tuple, Any

import numpy as np
import tensorflow_datasets as tfds
import os
from PIL import Image
from roborpc.collector.data_collector_utils import load_trajectory, crawler
from roborpc.data_convert.tfds_utils import MultiThreadedDatasetBuilder

tfds.core.utils.gcs_utils._is_gcs_disabled = True
DATA_PATH = "/media/jz08/SSD1/Log/droid/multi_task_dataset"

TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
VALID_DATA_PATH = os.path.join(DATA_PATH, 'valid')

print("Crawling all episode paths...")
episode_paths = crawler(DATA_PATH)
episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5')]
print(f"Found {len(episode_paths)} episodes!")
# print(episode_paths[0])
# get camera keys and robot keys
data_info = load_trajectory(os.path.join(episode_paths[0], 'trajectory.h5'))
camera_keys = [k for k in data_info[0]['observation'].keys() if 'camera' in k]
robot_keys = [k for k in data_info[0]['observation'].keys() if not ('camera' in k or 'timestamp' in k)]
IMAGE_RES = data_info[0]['observation'][camera_keys[0]]['color'].shape[:2]


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    def _resize_and_encode(image, size):
        image = Image.fromarray(image)
        return np.array(image.resize(size, resample=Image.BICUBIC))

    def _parse_example(episode_path):
        h5_filepath = os.path.join(episode_path, 'trajectory.h5')
        lang = str(episode_path.split(DATA_PATH)[-1].split('/')[0]).replace('_', '').replace('-', '')
        try:
            data = load_trajectory(h5_filepath)
        except (Exception,):
            print(f"Skipping trajectory because data couldn't be loaded for {episode_path}.")
            return None

        try:
            assert all(t.keys() == data[0].keys() for t in data)
            # for t in range(len(data)):
            #     for key in camera_keys:
            #         data[t]['observation'][key]['color'] = _resize_and_encode(
            #             data[t]['observation'][key]['color'], (IMAGE_RES[1], IMAGE_RES[0])
            #         )
            #         data[t]['observation'][key]['depth'] = _resize_and_encode(
            #             data[t]['observation'][key]['depth'], (IMAGE_RES[1], IMAGE_RES[0])
            #         )

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            episode_dict = {}
            for i, step in enumerate(data):
                obs = step['observation']
                action = step['action']

                episode_dict['observation'] = {}
                episode_dict['action'] = {}
                # episode_dict['action'] = {}
                for key in camera_keys:
                    episode_dict['observation'].update({key: obs[key]['color'][..., ::-1]})
                    # episode_dict['observation'][key].update({'depth': obs[key]['depth'][..., ::-1]})
                for key in robot_keys:
                    episode_dict['observation'].update(
                        {'joint_position': np.array([obs[key]['joint_position']]).reshape(-1),
                         'gripper_position': np.array(
                             [obs[key]['gripper_position']]).reshape(-1),
                         'cartesian_position': np.array(obs[key]['cartesian_position']).reshape(-1),
                         })
                    # episode_dict['action'].update(
                    #     {'joint_position': np.array([action[key]['joint_position']]).reshape(-1),
                    #      'gripper_position': np.array([action[key]['gripper_position']]).reshape(-1),
                    #      'cartesian_position': np.array(
                    #          [action[key]['cartesian_position']]).reshape(-1)})
                    episode_dict['action'] = np.concatenate([
                            # np.array([action[key]['joint_position']]).reshape(-1),
                            np.array(action[key]['cartesian_position']),
                            np.array(action[key]['gripper_position']),
                        ], axis=0).reshape(-1)

                episode_dict.update({
                    'discount': 1.0,
                    'reward': float((i == (len(data) - 1) and 'success' in episode_path)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': lang,
                })
                episode.append(episode_dict)
                episode_dict = {}
        except (Exception,):
            print(f"Skipping trajectory because there was an error in data processing for {episode_path}.")
            return None

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': h5_filepath,
            }
        }
        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)


class MultiTaskDatasetOcto(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    N_WORKERS = 10  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 100  # number of paths converted & stored in memory before writing to disk
    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...).
        H5 File
        dataset name: action
        - panda_1: Group
          - cartesian_position: (100, 6)
          - gripper_position: (100, 1)
          - joint_position: (100, 7)
        dataset name: observation
        - camera_1: Group
          - color: (100, 256, 256, 3)
          - depth: (100, 256, 256, 1)
        - camera_2: Group
          - color: (100, 256, 256, 3)
          - depth: (100, 256, 256, 1)
        - camera_3: Group
          - color: (100, 256, 256, 3)
          - depth: (100, 256, 256, 1)
        - panda_1: Group
          - gripper_position: (100, 1)
          - joint_position: (100, 7)
        - timestamp: Group
          - control: Group
            - control_start: (100,)
            - policy_start: (100,)
            - sleep_start: (100,)
            - step_end: (100,)
            - step_start: (100,)
        """

        # define dataset info
        obs_camera_info = {}
        for camera_key in camera_keys:
            obs_camera_info[camera_key] = tfds.features.Image(shape=IMAGE_RES + (3,), dtype=np.uint8,
                                                              encoding_format='jpeg')
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        **obs_camera_info,
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                        )}, doc=f'robot state'),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [6x cartesian position, 1x gripper position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    )
                }),
            }))

    def _split_paths(self):
        """Define data splits."""
        # create list of all examples -- by default we put all examples in 'train' split
        # add more elements to the dict below if you have more splits in your data

        return {
            'train': episode_paths,
        }

    def _split_train_valid_paths(self):
        """Define data splits."""
        # create list of all examples -- by default we put all examples in 'train' split
        # add more elements to the dict below if you have more splits in your data
        print("Crawling all episode paths...")
        train_episode_paths = crawler(TRAIN_DATA_PATH)
        train_episode_paths = [p for p in train_episode_paths if os.path.exists(p + '/trajectory.h5')]
        valid_episode_paths = crawler(VALID_DATA_PATH)
        valid_episode_paths = [p for p in valid_episode_paths if os.path.exists(p + '/trajectory.h5')]
        print(f"Found train {len(train_episode_paths)} episodes!")
        print(f"Found valid {len(valid_episode_paths)} episodes!")
        return {
            'train': train_episode_paths,
            'validation': valid_episode_paths,
        }
