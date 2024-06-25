import subprocess

from roborpc.robot_env import RobotEnv
from roborpc.controllers.composed_multi_controllers import ComposedMultiController
from roborpc.collector.data_collector import DataCollector

if __name__ == '__main__':
    try:
        controller_pid = subprocess.Popen(
            'bash -c "python /home/jz08/code_repo/roborpc/roborpc/controllers/multi_controllers.py"',
            shell=True)
        controller = ComposedMultiController()
        controller.connect_now()
        robot_env = RobotEnv()
        collector = DataCollector(env=robot_env, controller=controller)

        while True:
            try:
                print(controller.forward({"dynamixel_controller_left_arm":
                    {
                        "joint_positions": [0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 1.5707963267948966, 0.0],
                        "gripper_position": [0.0],
                    }}))
                collector.collect_trajectory()
            except KeyboardInterrupt:
                pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
                subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                break
    except Exception as e:
        print(e)
        pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
        subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])


