import os
import h5py
import numpy as np
from roborpc.collector.data_collector_utils import TrajectoryReader


#
# class TrajectoryReader:
#     def __init__(self, filepath):
#         self._hdf5_file = h5py.File(filepath, "r")
#         self._length = self.get_hdf5_length(self._hdf5_file)
#         self._video_readers = {}
#         self._index = 0
#
#     def length(self):
#         return self._length
#
#     def read_timestep(self, index=None, keys_to_ignore=[]):
#         if index is None:
#             index = self._index
#         assert index < self._length
#
#         timestep = self.load_hdf5_to_dict(self._hdf5_file, self._index, keys_to_ignore=[])
#         self._index += 1
#         return timestep
#
#     def get_hdf5_length(self, hdf5_file, keys_to_ignore=[]):
#         length = None
#
#         for key in hdf5_file.keys():
#             if key in keys_to_ignore:
#                 continue
#
#             curr_data = hdf5_file[key]
#             if isinstance(curr_data, h5py.Group):
#                 curr_length = self.get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
#             elif isinstance(curr_data, h5py.Dataset):
#                 curr_length = len(curr_data)
#             else:
#                 raise ValueError
#
#             if length is None:
#                 length = curr_length
#             assert curr_length == length
#
#         return length
#
#     def load_hdf5_to_dict(self, hdf5_file, index, keys_to_ignore=[]):
#         data_dict = {}
#
#         for key in hdf5_file.keys():
#             if key in keys_to_ignore:
#                 continue
#
#             curr_data = hdf5_file[key]
#             if isinstance(curr_data, h5py.Group):
#                 data_dict[key] = self.load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
#             elif isinstance(curr_data, h5py.Dataset):
#                 data_dict[key] = curr_data[index]
#             else:
#                 raise ValueError
#
#         return data_dict
#
#     def close(self):
#         self._hdf5_file.close()

