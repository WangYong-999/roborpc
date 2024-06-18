from __future__ import annotations

import contextlib
import os
import platform
import sys
import typing
from typing import IO, List, Optional, Tuple, cast

from loguru import logger as _loguru_logger
import psutil

from roborpc.common.context.context import context

if typing.TYPE_CHECKING:
    from loguru import Record

__all__ = ['DEFAULT_LOG_PATH', 'enable_execution_logger', 'logger']

_PROJECT_NAME = 'roborpc'
DEFAULT_LOG_PATH = os.path.join('/home', 'Log', _PROJECT_NAME)
if platform.machine() == 'x86_64':
    env_path = os.getenv('DEFAULT_LOG_PATH')
    if not env_path:
        DEFAULT_LOG_PATH = os.path.join(os.getenv('HOME', default='/tmp'), 'Log', _PROJECT_NAME)
    else:
        DEFAULT_LOG_PATH = env_path

_LOG_ROTATION_TIME_SPAN = '1 hour'
_LOG_RETENTION_TIME_SPAN = '1 month'


class _ErrorStreamToLogger:

    def __init__(self, *, name: Optional[str] = None) -> None:
        self._name = name
        self._buffer: List[str] = []

    def write(self, buffer: Optional[str]) -> None:
        if buffer is not None:
            self._buffer.append(buffer)

    def flush(self) -> None:
        if len(self._buffer) == 0:
            return
        pre_buffer = f'From {self._name}: ' if self._name is not None else ''
        info_str = pre_buffer + ''.join(self._buffer)
        info_level = 'ERROR' if ('error' in info_str or 'ERROR' in info_str or 'Error' in info_str) else 'INFO'
        try:
            _loguru_logger.opt(depth=1).log(info_level, info_str)
        except ValueError:
            # See https://loguru.readthedocs.io/en/stable/resources/recipes.html#using-loguru-s-logger-within-a-cython-module  # noqa: E501
            # and https://github.com/Delgan/loguru/issues/88
            _loguru_logger.patch(lambda record: record.update({
                'name': 'N/A',
                'function': '',
                'line': 0
            })).log(info_level, info_str)

        self._buffer = []


def _socket_only(record: Record) -> bool:
    return bool(record['module'] == 'socket_client')


def _arm_driver_feedback_only(record: Record) -> bool:
    return bool(record['module'] == 'arm_driver' and record['function'] == '_listen_feedback_handler')


def _get_ip() -> List[Tuple[str, str]]:
    """获取网卡名称和其ip地址，不包括回环."""
    ip_info = []
    info = psutil.net_if_addrs()
    for name, v in info.items():
        for item in v:  # type(item) == socket.snicaddr
            if item.family == 2:  # 2 == socket.AddressFamily.AF_INET
                if item.address != '127.0.0.1':
                    ip_info.append((name, item.address))
    return ip_info


def _log_format(_: Record) -> str:
    # add the thread name and the task id to log messages
    format_with_tid = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
                       '<level>{level: <8}</level> | '
                       '<magenta>{thread.name}</magenta> | '
                       f'<blue>{context.task_id}</blue> | '
                       '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n')

    return format_with_tid


class _Logger:

    def __init__(self, log_path: Optional[str] = None) -> None:
        # determine log file path and name
        if log_path is None:
            log_path = os.getenv('ROS_LOG_DIR')
        if not log_path:
            log_path = DEFAULT_LOG_PATH
        os.makedirs(log_path, exist_ok=True)
        try:
            self._log_filename_prefix = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        except (Exception, ):
            self._log_filename_prefix = _PROJECT_NAME
        self._log_path = log_path

        self._debug_file_pathname = os.path.join(log_path, 'debug', f'{self._log_filename_prefix}.debug.log')
        self._info_file_pathname = os.path.join(log_path, 'info', f'{self._log_filename_prefix}.info.log')

        _loguru_logger.remove()
        _loguru_logger.add(sys.__stdout__, format=_log_format)

        # add debug- and info-level loggers
        _loguru_logger.add(self._debug_file_pathname,
                           level='DEBUG',
                           rotation=_LOG_ROTATION_TIME_SPAN,
                           retention=_LOG_RETENTION_TIME_SPAN,
                           format=_log_format)
        _loguru_logger.add(self._info_file_pathname,
                           level='INFO',
                           rotation=_LOG_ROTATION_TIME_SPAN,
                           retention=_LOG_RETENTION_TIME_SPAN,
                           format=_log_format)

        _loguru_logger.info(f'Log files {self._debug_file_pathname} and {self._info_file_pathname} created.')

        # add stderr output to log.error (for third party packages' output)
        redirection = cast(IO[str], _ErrorStreamToLogger())
        contextlib.redirect_stderr(redirection).__enter__()

        self.logger = _loguru_logger

        self.logger.info(f'My ip is: {_get_ip()}')

    def enable_execution_logger(self) -> None:
        """Add socket and arm_driver_feedback loggers."""
        socket_file_pathname = os.path.join(self._log_path, 'socket', f'{self._log_filename_prefix}.socket.log')
        arm_driver_feedback_file_name = os.path.join(self._log_path, 'arm',
                                                     f'{self._log_filename_prefix}.arm_driver_feedback.log')

        # stdout(console) exclude the socket communication info
        _loguru_logger.remove()
        _loguru_logger.add(sys.__stdout__,
                           format=_log_format,
                           filter=lambda record: not ((_socket_only(record) and record['level'].no <= _loguru_logger.
                                                       level('INFO').no) or _arm_driver_feedback_only(record)))

        _loguru_logger.add(self._debug_file_pathname,
                           level='DEBUG',
                           filter=lambda record: not (_socket_only(record) or _arm_driver_feedback_only(record)),
                           rotation=_LOG_ROTATION_TIME_SPAN,
                           retention=_LOG_RETENTION_TIME_SPAN,
                           format=_log_format)
        _loguru_logger.add(self._info_file_pathname,
                           level='INFO',
                           filter=lambda record: not (_socket_only(record) or _arm_driver_feedback_only(record)),
                           rotation=_LOG_ROTATION_TIME_SPAN,
                           retention=_LOG_RETENTION_TIME_SPAN,
                           format=_log_format)

        _loguru_logger.add(socket_file_pathname,
                           level='INFO',
                           filter=_socket_only,
                           rotation=_LOG_ROTATION_TIME_SPAN,
                           retention=_LOG_RETENTION_TIME_SPAN,
                           format=_log_format)
        _loguru_logger.add(arm_driver_feedback_file_name,
                           level='INFO',
                           filter=_arm_driver_feedback_only,
                           rotation=_LOG_ROTATION_TIME_SPAN,
                           retention=_LOG_RETENTION_TIME_SPAN,
                           format=_log_format)

        _loguru_logger.info(f'Log files {socket_file_pathname} and {arm_driver_feedback_file_name} created.')


_logger_object = _Logger()


def enable_execution_logger() -> None:
    _logger_object.enable_execution_logger()


logger = _logger_object.logger
