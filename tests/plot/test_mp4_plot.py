import cv2
import matplotlib.pyplot as plt
import numpy as np

ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

# read mp4 file and extract frames and actions
cap = cv2.VideoCapture('F:\\papers\\roborpc\\figures\\square.mp4')
images = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    images.append(cv2.resize(np.array(frame), (256, 256)))

# exactly subsample the images to make the plot more readable
img_strip = np.concatenate(np.array(images[::20][::3]), axis=1)

# set up plt figure
figure_layout = [
    ['image'] * len(ACTION_DIM_LABELS),
    ACTION_DIM_LABELS
]
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplot_mosaic(figure_layout)
fig.set_size_inches([45, 10])

# plot actions
# pred_actions = np.array(pred_actions).squeeze()
# true_actions = np.array(true_actions).squeeze()
# for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
  # # actions have batch, horizon, dim, in this example we just take the first action for simplicity
  # axs[action_label].plot(pred_actions[:, 0, action_dim], label='predicted action')
  # axs[action_label].plot(true_actions[:, action_dim], label='ground truth')
  # axs[action_label].set_title(action_label)
  # axs[action_label].set_xlabel('Time in one episode')

axs['image'].imshow(img_strip)
axs['image'].set_xlabel('Time in one episode (subsampled)')
plt.legend()
plt.savefig('F:\\papers\\roborpc\\figures\\square_actions.png')