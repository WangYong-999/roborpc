import os
import subprocess
import sys
import traceback
from pathlib import Path
from subprocess import TimeoutExpired
from datetime import datetime

import http
import http.server
import json
from socketserver import ThreadingMixIn
import threading
import time
import requests

from roborpc.common.config_loader import config
from roborpc.common.logger_loader import logger


_TIMESTAMP_STRING_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
_CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith('/info'):
            return None
        else:
            self.send_error(404, 'Path not found')
            return None

    def do_HEAD(self, response_code: int = 200):
        self.send_response(response_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself

        json_message = json.loads(post_data.decode('utf-8'))
        command_code = str(json_message['command'])
        return_code = 200
        result_code = '0'
        result_msg = 'ok'

        if command_code == "startup_isaac":
            self.shutdown_isaac_sim()
            return_code = self.startup_isaac_sim()
        elif command_code == "shutdown_isaac":
            self.shutdown_isaac_sim()
        else:
            return_code = 404
            result_code = '404'
            result_msg = 'Service Not Found'

        return_message = {
            'command': command_code,
            'timestamp': datetime.now().strftime(_TIMESTAMP_STRING_FORMAT),
            'result_code': result_code,
            'result_msg': result_msg
        }

        self.do_HEAD(response_code=return_code)
        self.wfile.write(json.dumps(return_message).encode('utf-8'))

    isaac_process = None

    def startup_isaac_sim(self) -> int:
        logger.info("Startup isaac sim.")
        return_status_code = 200

        args = ''
        argv = sys.argv
        if len(argv) > 1:
            args = ' '.join(argv[1:])
        # startup isaac
        script_file_path = Path(_CURRENT_PATH).parent.parent / "scripts" / "run_sim_script.sh"
        py_file_path = Path(_CURRENT_PATH).parent.parent / "roborpc/sim_robots/isaac_sim" / "isaac_sim_task_runner.py"
        self.isaac_process = subprocess.Popen(["bash", str(script_file_path), str(py_file_path), args], cwd=_CURRENT_PATH)

        # wait for http server initialized
        def check_isaac_http_server() -> int:
            try:
                port = config['roborpc']['sim_robots']['isaac_sim']['task_port']
                response = requests.get(f'http://127.0.0.1:{port}/info')
                return response.status_code
            except requests.exceptions.ConnectionError:
                # logger.warning('Waiting for http server initialized: %s' % traceback.format_exc())
                return 0

        # wait 200s
        max_cnt = 200
        cnt = 0
        while cnt < max_cnt:
            time.sleep(1)
            cnt += 1
            status_code = check_isaac_http_server()
            if status_code == 200:
                break
        if cnt >= max_cnt:
            return_status_code = 500

        threading.Thread(target=self.check_isaac_sim_timeout, name='CheckIsaacTimeout', daemon=False).start()

        return return_status_code

    def check_isaac_sim_timeout(self):
        try:
            if self.isaac_process is not None:
                self.isaac_process.wait(500)
        except TimeoutExpired:
            logger.warning("Isaac timed out and exited. ")
            self.shutdown_isaac_sim()

    def shutdown_isaac_sim(self):
        logger.info("Shutdown isaac sim.")

        pid = subprocess.run(["pgrep", "-f", "task_runner"], capture_output=True)
        subprocess.run(["kill", "-9", *(pid.stdout.decode('utf-8').strip().rstrip().split('\n'))])
        time.sleep(1)
        self.isaac_process = None


class SimDaemonLauncher:
    def _listener_http(self) -> None:
        class ThreadingServer(ThreadingMixIn, http.server.HTTPServer):
            pass

        try:
            server = ThreadingServer((config['roborpc']['sim_robots']['daemon_host'],
                                      config['roborpc']['sim_robots']['daemon_port']),
                                     SimpleHTTPRequestHandler)
            server.serve_forever()
        except (Exception,):
            logger.error('Error in DaemonLauncher._listener_http: %s' % traceback.format_exc())

    def run(self):
        threading.Thread(target=self._listener_http, name='HttpListener', daemon=False).start()


if __name__ == '__main__':
    launcher = SimDaemonLauncher()
    launcher.run()
