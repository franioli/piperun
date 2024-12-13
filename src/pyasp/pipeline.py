import logging
import os
import time
from pathlib import Path
from typing import Any

import dask
from dask.distributed import Client, LocalCluster

from pyasp.shell import Command
from pyasp.steps import AspStepBase

logger = logging.getLogger("pyasp")


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with parallel steps: {self._steps}"

    def _setup_dask(self):
        """Set up the Dask distributed client."""
        cluster = LocalCluster(n_workers=self._workers)
        self._client = Client(cluster)
        logger.info(
            f"Dask client started with {self._workers or cluster.n_workers} workers."
        )

    def add_step(self, step: AspStepBase | Command | DelayedTask):
        """Add a step to the parallel block.

        Args:
            step: Pipeline step to add (must be AspStepBase or Command).

        Raises:
            TypeError: If step is not of allowed type.
        """
        if not isinstance(step, AspStepBase | Command | DelayedTask):
            raise TypeError(
                f"Invalid {step} in steps. Allowed steps are AspStepBase or Command."
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

            self.close()

        logger.info("Finished running the parallel block.\n")
        logger.info("---------------------------------------------\n")

    def close(self):
        """Shut down the Dask client."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Dask client shut down.")


class Pipeline:
    """A pipeline for executing sequential processing steps.

    Manages a sequence of processing steps that can include AspStepBase, Command,
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
            step: Step to add (must be DelayedTask, Command, ParallelBlock, Pipeline, or AspStepBase).

        Raises:
            TypeError: If step is not of allowed type.
        """
        if not isinstance(
            step, DelayedTask | Command | ParallelBlock | Pipeline | AspStepBase
        ):
            raise TypeError(
                f"Invalid {step} in steps. Allowed steps are AspStepBase, Command, Pipeline, or ParallelBlock."
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

        # config = {
        #     "steps": [
        #         OmegaConf.to_container(step, resolve=True) for step in self._steps
        #     ]
        # }
        # with open(filename, "w") as file:
        #     yaml.dump(config, file)

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

        # with open(filename, "r") as file:
        #     config = yaml.safe_load(file)

        # steps = [OmegaConf.create(step) for step in config["steps"]]
        # return cls(steps=steps)


if __name__ == "__main__":
    import tempfile

    from pyasp import steps
    from pyasp.spot5 import get_spot5_scenes

    data_dir = Path("demo/data")
    seed_dem = data_dir / "COP-DEM_GLO-30-DGED_2023_1_4326_ellipsoid.tif"

    # Write the oputpus to a temporary directory

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        scenes, images, rpc = get_spot5_scenes(
            image_dir=data_dir / "img", create_symlinks=True
        )

        # Test Pipeline
        pl = Pipeline()

        for f in rpc:
            pl.add_step(steps.AddSpotRPC(f))

        pl.add_step(
            steps.BundleAdjust(
                images=images,
                cameras=rpc,
                output_prefix="ba",
                t="rpc",
                heights_from_dem=seed_dem,
                heights_from_dem_uncertainty=30.0,
                max_iterations=500,
                ip_per_tile=150,
                matches_per_tile=50,
                datum="WGS_1984",
            )
        )

        mapproj_imgs = []
        for i, scene in scenes.items():
            image = scene / f"IMAGERY_{i}.TIF"
            model = scene / f"METADATA_{i}.DIM"
            output = temp_dir / f"mapproject_{i}.tif"
            pl.add_step(
                steps.MapProject(
                    dem=seed_dem,
                    camera_image=image,
                    camera_model=model,
                    output_image=output,
                    bundle_adjust_prefix="ba",
                    t="rpc",
                    tr=9.0123960e-05,
                    processes=8,
                    threads=18,
                )
            )
            mapproj_imgs.append(output)

        pl.add_step(
            steps.ParallelStereo(
                images=mapproj_imgs,
                cameras=rpc,
                dem=seed_dem,
                output_file_prefix="stereo",
                bundle_adjust_prefix="ba",
                alignment_method="none",
                stereo_algorithm="asp_sgm",
                t="rpcmaprpc",
                xcorr_threshold=2,
                cost_mode=3,
                corr_kernel=[9, 9],
                subpixel_mode=2,
                subpixel_kernel=[35, 35],
            )
        )
        pl.run()

        # Test ParallelBlock
        pb = ParallelBlock([steps.AddSpotRPC(f) for f in rpc], workers=2)
        pb.run()

        # Test Pipeline with ParallelBlock
        pl = Pipeline()
        pl.add_step(ParallelBlock([steps.AddSpotRPC(f) for f in rpc], workers=2))
        pl.run()
