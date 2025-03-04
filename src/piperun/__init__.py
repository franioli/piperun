import logging

from .pipeline import ParallelBlock, Pipeline
from .shell import Command, OutputCapture, run_command
from .tasks import DelayedTask
from .utils.logger import setup_logger

__version__ = "0.1.1"

__all__ = [
    "Pipeline",
    "Command",
    "OutputCapture",
    "run_command",
    "DelayedTask",
    "ParallelBlock",
    "setup_logger",
]

logger = setup_logger(level=logging.INFO, name="piperun")
