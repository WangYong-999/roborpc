import subprocess

from roborpc.robot_env import RobotEnv

if __name__ == '__main__':
    try:
        controller_pid = subprocess.Popen(
            'bash -c "python /home/jz08/code_repo/roborpc/roborpc/controllers/multi_controllers.py"',
            shell=True)
        robot_pid = subprocess.Popen(
            'bash -c "python /home/jz08/code_repo/roborpc/roborpc/robots/multi_robots.py"',
            shell=True)
        camera_pid = subprocess.Popen(
            'bash -c "python /home/jz08/code_repo/roborpc/roborpc/cameras/multi_cameras.py"',
            shell=True)
        robot_env = RobotEnv()
        controller = robot_env.controllers
        while True:
            try:
                obs = robot_env.get_observation()
                action = controller.forward(obs)
                print(action)
                # robot_env.step(action)
            except KeyboardInterrupt:
                controller_pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
                robot_pid = subprocess.run(["pgrep", "-f", "multi_robots"], capture_output=True)
                camera_pid = subprocess.run(["pgrep", "-f", "multi_cameras"], capture_output=True)
                subprocess.run(["kill", "-9", *(controller_pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                subprocess.run(["kill", "-9", *(robot_pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                subprocess.run(["kill", "-9", *(camera_pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
                break
    except Exception as e:
        print(e)
        controller_pid = subprocess.run(["pgrep", "-f", "multi_controllers"], capture_output=True)
        robot_pid = subprocess.run(["pgrep", "-f", "multi_robots"], capture_output=True)
        camera_pid = subprocess.run(["pgrep", "-f", "multi_cameras"], capture_output=True)
        subprocess.run(["kill", "-9", *(controller_pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
        subprocess.run(["kill", "-9", *(robot_pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
        subprocess.run(["kill", "-9", *(camera_pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
