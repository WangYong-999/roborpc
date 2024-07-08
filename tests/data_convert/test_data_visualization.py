from roborpc.collector.data_collector_utils import visualize_trajectory, replay_trajectory
from roborpc.robot_env import RobotEnv

robot_env = RobotEnv()

visualize_trajectory("/media/jz08/SSD1/Log/droid/data/success/2024-07-05/Fri_Jul__5_17:00:11_2024/trajectory.h5")
# replay_trajectory(env=robot_env,
#                   hdf5_filepath="/media/jz08/SSD1/Log/droid/data/success/2024-07-05/Fri_Jul__5_17:02:50_2024/trajectory.h5")
