import subprocess
import sys

from roborpc.robot_env import RobotEnv
from roborpc.collector.data_collector import DataCollector

if __name__ == '__main__':
    robot_env = RobotEnv()
    collector = DataCollector(env=robot_env)

    while True:
        try:
            collector.collect_trajectory(action_interpolation=True)
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            pid = subprocess.run(["pgrep", "-f", "test_collector"], capture_output=True)
            subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
            sys.exit(0)
