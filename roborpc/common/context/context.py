import functools
from typing import Callable, Optional, TypeVar

__all__ = ['context', 'update_context']


class Context:

    def __init__(self) -> None:
        self._task_id: Optional[int] = None

    @property
    def task_id(self) -> int:
        """Get task_id."""
        if self._task_id is None:
            return 0
        return self._task_id

    @task_id.setter
    def task_id(self, task_id: int) -> None:
        """Set task_id."""
        self._task_id = task_id


context = Context()

_TRequest = TypeVar('_TRequest')
_TResponse = TypeVar('_TResponse')
_TCaller = TypeVar('_TCaller', bound=object)


def update_context(
    f: Callable[[_TCaller, _TRequest, _TResponse],
                _TResponse]) -> Callable[[_TCaller, _TRequest, _TResponse], _TResponse]:
    """A decorator to update task id for the context."""

    @functools.wraps(f)
    def wrap(caller: _TCaller, request: _TRequest, response: _TResponse) -> _TResponse:
        # 重置task_id，避免残留上一次调用的值
        context.task_id = 0
        task_id = getattr(request, 'task_id', None)
        if task_id is not None:
            context.task_id = task_id
        result = f(caller, request, response)
        return result

    return wrap
