import os
import sys

from roborpc.collector.data_collector_utils import crawler
from roborpc.kinematics_solver.curobo_solver_kinematic import CuroboSolverKinematic
import h5py
from tqdm import tqdm

DATA_PATH = '/media/jz08/SSD1/Log/droid/multi_task_dataset'
robot_keys = ['panda_1']


kinematic_solver = CuroboSolverKinematic('panda')
episode_paths = crawler(DATA_PATH)
episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5')]
print(f"Total {len(episode_paths)} episodes found.")

for path in tqdm(episode_paths):
    # try:
    h5_filepath = os.path.join(path, 'trajectory.h5')
    hdf5_file = h5py.File(h5_filepath, "a")
    # update language in data_info
    if 'language' in hdf5_file['observation'].keys():
        hdf5_file.pop('/observation/language')
    if 'language' in hdf5_file['observation']['timestamp'].keys():
        hdf5_file.pop('/observation/timestamp/language')
    # name = str(path.split('multi_task_dataset/')[-1].split('/')[0]).replace('_','').replace('-','')
    # hdf5_file.create_group('/observation/')
    # hdf5_file['/observation/timestamp'].create_dataset('language', data=name, dtype=h5py.string_dtype(), shape=(1,))
    # update carteneous position in data_info
    for key in robot_keys:
        if 'cartesian_position' not in hdf5_file['observation'][key].keys():
            if key.startswith('realman'):
                robot_type = 'realman'
            elif key.startswith('panda'):
                robot_type = 'panda'
            else:
                raise ValueError(f"Unknown robot type {key}.")
            # print(f"{key}: {hdf5_file['observation'][key]['joint_position'][()].tolist()}")
            cartesian_position = kinematic_solver.forward_batch_kinematics(joint_angles={key: hdf5_file['observation'][key]['joint_position'][()].tolist()})[key]
            # hdf5_file.pop(f"/observation/{key}/cartesian_position")
            hdf5_file.create_dataset(f"/observation/{key}/cartesian_position", data=cartesian_position)
    # double check
    # print(hdf5_file['observation'][key]['joint_position'][()][2])
    # print(hdf5_file['observation'][key]['cartesian_position'][()][2])
    # s = kinematic_solver.forward_kinematics(
    #     {key: hdf5_file['observation'][key]['joint_position'][()][2].tolist()})
    # print(s)
    hdf5_file.close()
    # except Exception as e:
    #     print(f"Error in {path}: {e}")


