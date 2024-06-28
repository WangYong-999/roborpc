import subprocess

from roborpc.robot_env import RobotEnv
from roborpc.controllers.composed_multi_controllers import ComposedMultiController
from roborpc.collector.data_collector import DataCollector

if __name__ == '__main__':
    try:
        controller_pid = subprocess.Popen(
            'bash -c "python /home/jz08/code_repo/roborpc/roborpc/controllers/multi_controllers.py"',
            shell=True)
        robot_env = RobotEnv()
        collector = DataCollector(env=robot_env)

        while True:
            try:
                collector.collect_trajectory()
            except KeyboardInterrupt:
                pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
                subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                break
    except Exception as e:
        print(e)
        pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
        subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])


