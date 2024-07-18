import subprocess
import sys

import cv2

from roborpc.robot_env import RobotEnv
from roborpc.cameras.composed_multi_cameras import ComposedMultiCameras
from roborpc.collector.data_collector import DataCollector


if __name__ == '__main__':
    # camera_1_pid = subprocess.Popen(
    #     'bash -c "python /home/jz08/code_repo/roborpc/roborpc/cameras/multi_cameras.py'
    #     ' --rpc_port 4247 --camera_ids realsense_camera_1"',
    #     shell=True)
    # camera_2_pid = subprocess.Popen(
    #     'bash -c "python /home/jz08/code_repo/roborpc/roborpc/cameras/multi_cameras.py'
    #     ' --rpc_port 4248 --camera_ids realsense_camera_2"',
    #     shell=True)
    camera = ComposedMultiCameras()
    camera.connect_now()
    print(camera.get_camera_intrinsics())

    while True:
        try:
            camera_info = camera.read_camera()
            for camera_id, info in camera_info.items():
                cv2.imshow(camera_id, info['color'])
                cv2.waitKey(1)
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            multi_cameras_pid = subprocess.run(["pgrep", "-f", "multi_cameras"], capture_output=True)
            subprocess.run(["kill", "-9", *(multi_cameras_pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
            cv2.destroyAllWindows()
            sys.exit(0)

