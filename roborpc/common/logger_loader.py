from __future__ import annotations

import contextlib
import os
import sys
import typing
from typing import IO, List, Optional, Tuple, cast

from loguru import logger as _loguru_logger
import psutil


if typing.TYPE_CHECKING:
    from loguru import Record


_PROJECT_NAME = 'roborpc'
_DEFAULT_LOG_PATH = os.path.join(os.getenv('HOME', default='/tmp'), 'Log', _PROJECT_NAME)


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
        try:
            _loguru_logger.opt(depth=1).error(info_str)
        except ValueError:
            # See https://loguru.readthedocs.io/en/stable/resources/recipes.html#using-loguru-s-logger-within-a-cython-module  # noqa: E501
            # and https://github.com/Delgan/loguru/issues/88
            _loguru_logger.patch(lambda record: record.update({
                'name': 'N/A',
                'function': '',
                'line': 0
            })).error(info_str)

        self._buffer = []


def _get_ip() -> List[Tuple[str, str]]:
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
                       '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n')

    return format_with_tid


class _Logger:

    def __init__(self, log_path: Optional[str] = None) -> None:
        # determine log file path and name
        if log_path is None:
            log_path = os.getenv('ROS_LOG_DIR')
        if not log_path:
            log_path = _DEFAULT_LOG_PATH
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
                           rotation='1 day',
                           retention='1 month',
                           format=_log_format)
        _loguru_logger.add(self._info_file_pathname,
                           level='INFO',
                           rotation='1 day',
                           retention='1 month',
                           format=_log_format)

        _loguru_logger.info(f'Log files {self._debug_file_pathname} and {self._info_file_pathname} created.')

        # add stderr output to log.error (for third party packages' output)
        # redirection = cast(IO[str], _ErrorStreamToLogger())
        # contextlib.redirect_stderr(redirection).__enter__()

        self.logger = _loguru_logger

        self.logger.info(f'My ip is: {_get_ip()}')


logger = _Logger().logger
