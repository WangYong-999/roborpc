from typing import Iterator, Tuple, Any

import numpy as np
import tensorflow_datasets as tfds
import os
from PIL import Image

from roborpc.collector.data_collector_utils import load_trajectory, crawler
from roborpc.data_convert.tfds_utils import MultiThreadedDatasetBuilder

LANGUAGE_INSTRUCTION = "close the box"
DATA_PATH = "/media/jz08/SSD1/Log/droid/multi_task_dataset/Turn the handle and open the door"

TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
VALID_DATA_PATH = os.path.join(DATA_PATH, 'valid')

# (180, 320) is the default resolution, modify if different resolution is desired
IMAGE_RES = (180, 320)


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    def _resize_and_encode(image, size):
        image = Image.fromarray(image)
        return np.array(image.resize(size, resample=Image.BICUBIC))

    def _parse_example(episode_path):
        h5_filepath = os.path.join(episode_path, 'trajectory.h5')
        kinematic_solver = {}
        try:
            data = load_trajectory(h5_filepath)
        except (Exception,):
            print(f"Skipping trajectory because data couldn't be loaded for {episode_path}.")
            return None

        # get language instruction -- modify if more than one instruction
        lang = LANGUAGE_INSTRUCTION

        try:
            assert all(t.keys() == data[0].keys() for t in data)
            camera_keys = [k for k in data[0]['observation'].keys() if 'camera' in k]
            robot_keys = [k for k in data[0]['observation'].keys() if not ('camera' in k or 'timestamp' in k)]
            # for t in range(len(data)):
            #     for key in camera_keys:
            #         data[t]['observation'][key]['color'] = _resize_and_encode(
            #             data[t]['observation'][key]['color'], (IMAGE_RES[1], IMAGE_RES[0])
            #         )

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            episode_dict = {}
            for key in robot_keys:
                if 'cartesian_position' not in data[0]['observation'][key].keys():
                    if key.startswith('realman'):
                        kinematic_solver['realman'] = CuroboSolverKinematic('realman')
                    elif key.startswith('panda'):
                        kinematic_solver['panda'] = CuroboSolverKinematic('panda')

            for i, step in enumerate(data):
                obs = step['observation']
                action = step['action']

                episode_dict['observation'] = {}
                episode_dict['action'] = {}
                # episode_dict['action'] = {}
                for key in camera_keys:
                    episode_dict['observation'][key] = {}
                    episode_dict['observation'][key].update({'color': obs[key]['color'][..., ::-1]})
                    episode_dict['observation'][key].update({'depth': obs[key]['depth'][..., ::-1]})

                for key in robot_keys:
                    if 'cartesian_position' not in obs[key].keys():
                        if key.startswith('realman'):
                            robot_type = 'realman'
                        elif key.startswith('panda'):
                            robot_type = 'panda'
                        else:
                            raise ValueError(f"Unknown robot type {key}.")
                        cartesian_position = kinematic_solver[robot_type].forward_kinematics(joint_angles=
                                                                                             {key: obs[key][
                                                                                                 'joint_position'].tolist()})[
                            key]
                    else:
                        cartesian_position = obs[key]['cartesian_position']
                    episode_dict['observation'][key] = {}
                    episode_dict['action'][key] = {}
                    episode_dict['observation'][key].update({'joint_position': np.array([obs[key]['joint_position']]),
                                                             'gripper_position': np.array(
                                                                 [obs[key]['gripper_position']]),
                                                             'cartesian_position': np.array(cartesian_position),
                                                             })
                    episode_dict['action'][key].update({'joint_position': np.array([action[key]['joint_position']]),
                                                        'gripper_position': np.array([action[key]['gripper_position']]),
                                                        'cartesian_position': np.array(
                                                            [action[key]['cartesian_position']])})

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


class MultiTaskDataset(MultiThreadedDatasetBuilder):
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
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'exterior_image_1_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 1 left viewpoint',
                        ),
                        'exterior_image_2_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 2 left viewpoint'
                        ),
                        'wrist_image_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB left viewpoint',
                        ),
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Robot Cartesian state',
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Gripper position state',
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Joint position state'
                        )
                    }),
                    'action': tfds.features.FeaturesDict({
                        'panda_1': tfds.features.FeaturesDict({
                            'cartesian_position': tfds.features.Tensor(
                                shape=(6,),
                                dtype=np.float64,
                                doc='Commanded Cartesian position'
                            ),
                            'gripper_position': tfds.features.Tensor(
                                shape=(1,),
                                dtype=np.float64,
                                doc='Commanded gripper position'
                            ),
                            'joint_position': tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float64,
                                doc='Commanded joint position'
                            ),
                        }),
                    }),
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
                    ),
                    'recording_folderpath': tfds.features.Text(
                        doc='Path to the folder of recordings.'
                    )
                }),
            }))

    def _split_paths(self):
        """Define data splits."""
        # create list of all examples -- by default we put all examples in 'train' split
        # add more elements to the dict below if you have more splits in your data
        print("Crawling all episode paths...")
        episode_paths = crawler(DATA_PATH)
        episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5')]
        print(f"Found {len(episode_paths)} episodes!")
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
