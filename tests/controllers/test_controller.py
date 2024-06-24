import subprocess
from roborpc.controllers.composed_multi_controllers import ComposedMultiController


if __name__ == '__main__':
    try:
        pid = subprocess.Popen('bash -c "python /home/jz08/code_repo/roborpc/roborpc/controllers/multi_controllers.py"', shell=True)
        controller = ComposedMultiController()
        result = controller.connect_now()
        print(result)
        print("====================")
        for r in result.values():
            if not r:
                raise Exception("Failed to connect to all controllers")
        while True:
            try:
                print(controller.forward({"dynamixel_controller_left_arm":
                                              {
                                                  "robot_state": {
                                                      "joint_positions": [0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0],
                                                      "gripper_position": 0.0,
                                                  }
                                              }}))
            except KeyboardInterrupt:
                pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
                subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                break
    except Exception as e:
        print(e)
        pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
        subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
