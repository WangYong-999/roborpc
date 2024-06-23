import os
from cv2 import aruco

# Robot Params #
nuc_ip = "192.168.1.101"
robot_ip = "192.168.1.100"
laptop_ip = "192.168.1.104"
sudo_password = "eai"
robot_type = "panda"  # 'panda' or 'fr3'
robot_serial_number = "295341-1324807"

# Camera ID's #
hand_camera_id = "18361939"
varied_camera_1_id = "23282896" #left
varied_camera_2_id = "23343100"

# hand_camera_id = "10805454_left"
# varied_camera_1_id = "21729895_left" #left
# varied_camera_2_id = "29392465"

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5
CHARUCOBOARD_CHECKER_SIZE = 0.032
CHARUCOBOARD_MARKER_SIZE = 0.020
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
# CHARUCOBOARD_ROWCOUNT = 3
# CHARUCOBOARD_COLCOUNT = 4
# CHARUCOBOARD_CHECKER_SIZE = 0.040
# CHARUCOBOARD_MARKER_SIZE = 0.030
# ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = ""

# Code Version [DONT CHANGE] #
droid_version = "1.3"

