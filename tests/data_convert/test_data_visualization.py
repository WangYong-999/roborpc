from roborpc.collector.data_collector_utils import visualize_trajectory, replay_trajectory
from roborpc.robot_env import RobotEnv

robot_env = RobotEnv()

# visualize_trajectory("/home/jz08/Log/droid/data/success/2024-07-03/Wed_Jul__3_18:56:34_2024/trajectory.h5")
replay_trajectory(env=robot_env,
                  hdf5_filepath="/home/jz08/Log/droid/data/success/2024-07-04/Thu_Jul__4_09:19:14_2024/trajectory.h5")
