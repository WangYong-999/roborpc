# read robot state
import numpy as np

robot_obs_list = np.load('/home/roboticlab/yong/roborpc/ours_robot_obs_list.npy')
print(list(robot_obs_list[-1]))

# robot_obs_list  shape (500, 7)

# plot the robot state
import matplotlib.pyplot as plt

plt.plot(robot_obs_list[:,0], label='x')
plt.plot(robot_obs_list[:,1], label='y')
plt.plot(robot_obs_list[:,2], label='z')
plt.plot(robot_obs_list[:,3], label='roll')
plt.plot(robot_obs_list[:,4], label='pitch')
plt.plot(robot_obs_list[:,5], label='yaw')
plt.legend()
plt.show()