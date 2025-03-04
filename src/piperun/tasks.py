import logging
import time
from typing import Any

import dask

logger = logging.getLogger("piperun")


class DelayedTask:
    """A class to handle delayed task execution using Dask.

    This class wraps a callable task and provides utilities for delayed execution,
    timing measurement, and visualization using Dask's delayed functionality.

    Attributes:
        _elapsed_time (float): Time taken for the last task execution in seconds.
        _task (dask.delayed): The wrapped Dask delayed task object.

    Example:
        >>> def my_task(x):
        ...     return x * 2
        >>> delayed = DelayedTask(my_task, 5)
        >>> result = delayed.compute()
    """

    _elaspsed_time = None
    _task = None

    def __init__(
        self,
        task: callable,
        *args,
        **kwargs,
    ):
        """Initialize a DelayedTask instance.

        Args:
            task (callable): The function to be executed.
            *args: Variable length argument list for the task.
            **kwargs: Arbitrary keyword arguments for the task.

        Raises:
            ValueError: If task is not callable.
        """
        if not callable(task):
            raise ValueError(
                "Invalid task argument. Task must be a callable function (or a dask delayed object)."
            )
        self._task = dask.delayed(task)(*args, **kwargs)

    def __repr__(self):
        """Return string representation of the DelayedTask.

        Returns:
            str: String representation including class name and task.
        """
        return f"{self.__class__.__name__}: {self._task}"

    @property
    def elapsed_time(self) -> float:
        """Get the elapsed time of the last command execution.

        Returns:
            float: The elapsed time in seconds. None if not executed yet.
        """
        if self._elaspsed_time is None:
            logger.info("The step has not been executed yet.")
        return self._elaspsed_time

    def compute(self) -> Any:
        """Execute the delayed task and measure execution time.

        Returns:
            Any: Result of the computed task.
        """
        start_time = time.perf_counter()
        ret = self._task.compute()
        self._elaspsed_time = time.perf_counter() - start_time
        logger.info(
            f"Command {self.__class__.__name__} took {self._elaspsed_time:.4f} s."
        )
        return ret

    def run(self):
        """Alias for compute() method.

        Returns:
            Any: Result of the computed task.
        """
        return self.compute()

    def visualize(self, filename):
        """Visualize the task graph using Dask's visualization.

        Args:
            filename (str): Path where the visualization will be saved.
        """
        self._task.visualize(filename)
