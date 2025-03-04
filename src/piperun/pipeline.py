import logging
import os
from pathlib import Path
from typing import Any

import dask
from dask.distributed import Client, LocalCluster

from piperun.shell import Command
from piperun.tasks import DelayedTask

logger = logging.getLogger("piperun")


class Pipeline:
    """A pipeline for executing sequential processing steps.

    Manages a sequence of processing steps that can include Command or DelayedTask objects,
    ParallelBlock, or nested Pipeline objects.

    Attributes:
        _steps (List[Any]): List of pipeline steps to execute.
    """

    _steps: list[Any] = []

    def __init__(self, steps: list[Any] | dict[str, Any] = None):
        if steps is None:
            steps = []
        """Initialize a Pipeline instance.

        Args:
            steps: List or dictionary of pipeline steps to execute.
        """
        self._steps = []  #  Initialize an empty list of steps

        if isinstance(steps, dict):
            steps = [step for step in steps.values()]

        for step in steps:
            self.add_step(step)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with steps: {self.steps}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with {len(self.steps)} steps."

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, key: int) -> Any:
        if key >= len(self.steps):
            raise IndexError(
                f"Invalid step number {key}. Must be less than {len(self.steps)}."
            )
        return self.steps[key]

    @property
    def steps(self) -> list[Any]:
        """Get the list of pipeline steps.

        Returns:
            List of pipeline steps.
        """
        return self._steps

    def add_step(self, step: Any):
        """Add a step to the pipeline.

        Args:
            step: Step to add (must be DelayedTask, Command, ParallelBlock, Pipeline).

        Raises:
            TypeError: If step is not of allowed type.
        """
        if not isinstance(step, DelayedTask | Command | ParallelBlock | Pipeline):
            raise TypeError(
                f"Invalid {step} in steps. Allowed steps are DelayedTask, Command, Pipeline, or ParallelBlock."
            )
        self._steps.append(step)

    def remove_step(self, step_number: int):
        """Remove a step from the pipeline by its index.

        Args:
            step_number: Index of the step to remove.

        Raises:
            IndexError: If step_number is out of range.
        """
        if step_number >= len(self._steps):
            raise IndexError(
                f"Invalid step number {step_number}. Must be less than {len(self._steps)}."
            )
        self._steps.pop(step_number)
        logger.info(f"Removed step {step_number} from the pipeline.\n")

    def replace_step(self, step_number: int, new_step: Any):
        """Replace a step in the pipeline with a new one.

        Args:
            step_number: Index of the step to replace.
            new_step: New step to insert at the specified index.

        Raises:
            IndexError: If step_number is out of range.
        """
        if step_number >= len(self._steps):
            raise IndexError(
                f"Invalid step number {step_number}. Must be less than {len(self._steps)}."
            )
        self._steps[step_number] = new_step
        logger.info(f"Replaced step {step_number} with {new_step}.\n")

    def clear(self):
        """Remove all steps from the pipeline."""
        self._steps = []
        logger.info("Cleared all steps from the pipeline.\n")

    def run(self):
        """Run all steps in the pipeline sequentially."""
        logger.info("Starting the stereo pipeline...\n")
        for step in self._steps:
            logger.info(f"Running step: {step}")
            step.run()
            logger.info(f"Finished step: {step}")
            logger.info("---------------------------------------------\n")

        logger.info("Finished running the stereo pipeline.\n")
        logger.info("---------------------------------------------\n")

    def run_step(self, step_number: int):
        """Run a specific step in the pipeline by its index.

        Args:
            step_number: Index of the step to run.

        Raises:
            IndexError: If step_number is out of range.
        """
        if step_number >= len(self._steps):
            raise IndexError(
                f"Invalid step number {step_number}. Must be less than {len(self._steps)}."
            )

        logger.info(f"Running step {step_number}...\n")
        step = self._steps[step_number]
        logger.info(f"Running step: {step}")
        step.run()
        logger.info(f"Finished step: {step}.")

    def run_from_step(self, step_number: int):
        """Run the pipeline starting from a specific step.

        Args:
            step_number: Index of the step to start from.

        Raises:
            IndexError: If step_number is out of range.
        """
        if step_number >= len(self._steps):
            raise IndexError(
                f"Invalid step number {step_number}. Must be less than {len(self._steps)}."
            )

        logger.info(f"Resuming the stereo pipeline from step {step_number}...\n")
        for i, step in enumerate(self._steps):
            if i < step_number:
                logger.info(f"Skipping step: {step}")
                continue
            logger.info(f"Running step: {step}")
            step.run()
            logger.info(f"Finished step: {step}.")
            logger.info("---------------------------------------------\n")

        logger.info("Finished running the stereo pipeline.\n")
        logger.info("---------------------------------------------\n")

    def run_until_step(self, step_number: int):
        """Run the pipeline until a specific step is completed.

        Args:
            step_number: Index of the step to run until. The pipeline will run up to this step (not including).

        Raises:
            IndexError: If step_number is out of range.
        """
        if step_number >= len(self._steps):
            raise IndexError(
                f"Invalid step number {step_number}. Must be less than {len(self._steps)}."
            )

        logger.info(f"Running the stereo pipeline until step {step_number}...\n")
        for i, step in enumerate(self._steps):
            if i == step_number:
                break
            logger.info(f"Running step: {step}")
            step.run()
            logger.info(f"Finished step: {step}.")
            logger.info("---------------------------------------------\n")

        logger.info("Finished running the stereo pipeline.\n")
        logger.info("---------------------------------------------\n")

    def to_yaml(self, filename: Path):
        """Save the pipeline configuration to a YAML file.

        Args:
            filename: Path to save the YAML file.

        Raises:
            NotImplementedError: Method not implemented yet.
        """
        raise NotImplementedError("This method is not implemented yet.")

    @classmethod
    def from_yaml(cls, filename: Path):
        """Load a pipeline configuration from a YAML file.

        Args:
            filename: Path to the YAML configuration file.

        Returns:
            Pipeline: New pipeline instance configured from the YAML file.

        Raises:
            NotImplementedError: Method not implemented yet.
        """
        raise NotImplementedError("This method is not implemented yet.")


class ParallelBlock:
    """A parallel block of pipeline steps using Dask for distributed execution.

    Allows running multiple pipeline steps in parallel using Dask distributed computing.

    Attributes:
        _steps (list): List of pipeline steps to be executed in parallel.
        _workers (int): Number of Dask workers to use.
        _client (Client): Dask distributed client instance.
    """

    _steps = []

    def __init__(
        self,
        steps: list[Any] | dict[str, Any] = None,
        workers: int = None,
    ):
        """Initialize a ParallelBlock instance.

        Args:
            steps: List or dictionary of pipeline steps to run in parallel.
            workers: Number of Dask workers to use. If None, uses all available cores.
        """

        self._steps = []  # Initialize an empty list of steps

        if steps is None:
            steps = []
        elif isinstance(steps, dict):
            steps = [step for step in steps.values()]

        for step in steps:
            self.add_step(step)

        self._workers = workers
        self._client = None

    def __enter__(self):
        """Enter the context manager, initializing the Dask cluster and client."""
        self._setup_dask()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, closing the Dask client and cluster."""
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with parallel steps: {self._steps}"

    def _setup_dask(self):
        """Set up the Dask distributed client."""
        cluster = LocalCluster(n_workers=self._workers)
        self._client = Client(cluster)
        logger.info(
            f"Dask client started with {self._workers or cluster.n_workers} workers."
        )

    def add_step(self, step: Command | DelayedTask):
        """Add a step to the parallel block.

        Args:
            step: Pipeline step to add (must be Command or DelayedTask).

        Raises:
            TypeError: If step is not of allowed type.
        """
        if not isinstance(step, Command | DelayedTask):
            raise TypeError(
                f"Invalid {step} in steps. Allowed steps are Command or DelayedTask."
            )
        self._steps.append(step)

    def run(self, parallel_count: int = None):
        """
        Run the steps in parallel using Dask.

        Args:
            parallel_count: Number of steps to run in parallel. If None, uses number of Dask workers.
        """

        def _run_step(step):
            logger.info(f"Running step: {step}")
            step.run()
            return step

        if not self._steps:
            logger.info("No steps to run in the parallel block.")
            return

        if parallel_count:
            self._workers = parallel_count
        elif self._workers is None:
            self._workers = min(os.cpu_count(), len(self._steps))

        logger.info(f"Starting parallel execution with {self._workers} processes...")

        delayed_tasks = [step for step in self._steps if isinstance(step, DelayedTask)]
        non_delayed_tasks = [
            step for step in self._steps if not isinstance(step, DelayedTask)
        ]

        if delayed_tasks:
            # Use Dask to manage parallelization for DelayedTasks
            delayed_results = [task._task for task in delayed_tasks]
            dask.compute(*delayed_results)
        if non_delayed_tasks:
            # Use Dask distributed for non-DelayedTask steps

            if self._client is None:
                self._setup_dask()

            futures = []
            for i in range(0, len(non_delayed_tasks), self._workers):
                batch = non_delayed_tasks[i : i + self._workers]
                futures.extend(self._client.map(_run_step, batch))
            results = self._client.gather(futures)

            for result in results:
                logger.info(f"Completed step: {result}")

        logger.info("Finished running the parallel block.\n")
        logger.info("---------------------------------------------\n")

    def close(self):
        """Shut down the Dask client."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Dask client shut down.")
