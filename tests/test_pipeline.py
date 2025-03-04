import time

import pytest

from piperun.pipeline import DelayedTask, ParallelBlock, Pipeline


def add(a, b):
    """Test helper function"""
    time.sleep(0.1)  # Simulate work
    return a + b


def slow_task(x):
    """Helper function that simulates slow processing"""
    time.sleep(0.1)
    return x * 2


def multiply(x):
    """Test helper function"""
    time.sleep(0.1)  # Simulate work
    return x * 2


""" Tests for DelayedTask class """


@pytest.fixture
def basic_task():
    """Fixture for a simple DelayedTask"""
    return DelayedTask(add, 2, 3)


def test_delayed_task_creation():
    """Test DelayedTask initialization"""
    task = DelayedTask(add, 1, 2)
    assert isinstance(task, DelayedTask)
    assert task.elapsed_time is None


def test_delayed_task_invalid_creation():
    """Test DelayedTask creation with invalid input"""
    with pytest.raises(ValueError):
        DelayedTask("not_callable")


def test_delayed_task_compute():
    """Test compute() method with basic arithmetic"""
    task = DelayedTask(add, 2, 3)
    result = task.compute()
    assert result == 5
    assert task.elapsed_time > 0


def test_delayed_task_run():
    """Test run() method (alias for compute)"""
    task = DelayedTask(multiply, 4)
    result = task.run()
    assert result == 8
    assert task.elapsed_time > 0


def test_delayed_task_chaining():
    """Test chaining multiple delayed tasks"""
    task1 = DelayedTask(add, 2, 3)
    result1 = task1.compute()
    task2 = DelayedTask(multiply, result1)
    result2 = task2.compute()
    assert result2 == 10


def test_delayed_task_elapsed_time(basic_task):
    """Test elapsed_time property behavior"""
    assert basic_task.elapsed_time is None
    basic_task.compute()
    assert basic_task.elapsed_time > 0
    assert isinstance(basic_task.elapsed_time, float)


def test_delayed_task_repr(basic_task):
    """Test string representation"""
    repr_str = repr(basic_task)
    assert "DelayedTask" in repr_str
    assert "add" in repr_str.lower()


def test_delayed_task_with_kwargs():
    """Test DelayedTask with keyword arguments"""

    def func_with_kwargs(x, y, multiplier=1):
        return (x + y) * multiplier

    task = DelayedTask(func_with_kwargs, 2, 3, multiplier=2)
    result = task.compute()
    assert result == 10


@pytest.mark.parametrize(
    "inputs,expected", [((2, 3), 5), ((0, 0), 0), ((-1, 1), 0), ((10, 20), 30)]
)
def test_delayed_task_with_different_inputs(inputs, expected):
    """Test delayed task with different input combinations"""
    task = DelayedTask(add, *inputs)
    result = task.compute()
    assert result == expected


""" Tests for ParallBlock class """


@pytest.fixture
def parallel_block():
    """Fixture providing a basic ParallelBlock"""
    return ParallelBlock()


@pytest.fixture
def populated_block():
    """Fixture providing a ParallelBlock with preset tasks"""
    block = ParallelBlock()
    for i in range(5):
        block.add_step(DelayedTask(slow_task, i))
    return block


def test_parallel_block_creation():
    """Test ParallelBlock initialization"""
    block = ParallelBlock()
    assert len(block._steps) == 0
    assert block._client is None


def test_add_steps():
    """Test adding steps to ParallelBlock"""
    block = ParallelBlock()
    task = DelayedTask(slow_task, 1)
    block.add_step(task)
    assert len(block._steps) == 1


def test_parallel_execution(populated_block):
    """Test parallel execution of tasks"""
    start_time = time.perf_counter()
    populated_block.run(parallel_count=5)
    duration = time.perf_counter() - start_time

    # With 5 tasks, sleep 0.1s each, run in parallel with 2 workers
    # Should take ~0.3s (3 batches) instead of 0.5s (sequential)
    assert duration < 0.4  # Allow some overhead


def test_client_cleanup(populated_block):
    """Test client cleanup after execution"""
    populated_block.run(parallel_count=2)
    assert populated_block._client is None


def test_empty_block_execution():
    """Test running an empty parallel block"""
    block = ParallelBlock()
    block.run()  # Should not raise any errors


@pytest.mark.parametrize(
    "parallel_count,expected_duration",
    [
        (1, 0.5),  # Sequential - all 5 tasks
        (5, 0.1),  # Fully parallel
        (2, 0.3),  # 3 batches
    ],
)
def test_different_parallel_counts(populated_block, parallel_count, expected_duration):
    """Test different parallel execution configurations"""
    start_time = time.perf_counter()
    populated_block.run(parallel_count=parallel_count)
    duration = time.perf_counter() - start_time
    assert duration < expected_duration + 1  # Allow overhead


def test_error_handling():
    """Test error handling during parallel execution"""

    def failing_task():
        raise ValueError("Task failed")

    block = ParallelBlock()
    block.add_step(DelayedTask(failing_task))

    with pytest.raises(Exception):
        block.run()
    assert block._client is None  # Client should be cleaned up even after error


def test_mixed_task_types():
    """Test parallel block with different types of tasks"""
    block = ParallelBlock()
    block.add_step(DelayedTask(slow_task, 1))
    block.add_step(DelayedTask(lambda x: x + 1, 2))
    block.run(parallel_count=2)
    assert block._client is None


def test_large_batch_execution():
    """Test execution with large number of tasks"""
    block = ParallelBlock()
    for i in range(20):
        block.add_step(DelayedTask(slow_task, i))

    start_time = time.perf_counter()
    block.run(parallel_count=4)
    duration = time.perf_counter() - start_time

    # With 20 tasks and 4 workers, should take ~0.5s (5 batches)
    assert duration < 0.6  # Allow overhead


""" Tests for Pipeline class """


@pytest.fixture
def empty_pipeline():
    """Fixture providing an empty Pipeline"""
    return Pipeline()


@pytest.fixture(scope="function")
def basic_pipeline(basic_task):
    """Fixture providing a Pipeline with basic tasks.
    Creates a new pipeline instance for each test."""
    pipe = Pipeline()
    for _ in range(3):
        pipe.add_step(basic_task)
    return pipe


def test_pipeline_creation(empty_pipeline):
    """Test Pipeline initialization"""
    assert len(empty_pipeline) == 0
    assert isinstance(empty_pipeline.steps, list)


def test_pipeline_with_dict_init(basic_task):
    """Test Pipeline initialization with dictionary"""
    steps = {"step1": basic_task, "step2": basic_task}
    pipe = Pipeline(steps)
    assert len(pipe) == 2


def test_add_step(empty_pipeline, basic_task):
    """Test adding steps to Pipeline"""
    empty_pipeline.add_step(basic_task)
    empty_pipeline.add_step(basic_task)
    assert len(empty_pipeline) == 2
    assert empty_pipeline[-1] == basic_task


def test_remove_step(basic_pipeline):
    """Test removing steps from Pipeline"""
    initial_len = len(basic_pipeline)
    basic_pipeline.remove_step(0)
    assert len(basic_pipeline) == initial_len - 1


def test_replace_step(basic_pipeline):
    """Test replacing steps in Pipeline"""
    new_task = DelayedTask(slow_task, 10)
    basic_pipeline.replace_step(0, new_task)
    assert basic_pipeline[0] == new_task


def test_clear_pipeline(basic_pipeline):
    """Test clearing all steps from Pipeline"""
    basic_pipeline.clear()
    assert len(basic_pipeline) == 0


def test_run_pipeline(basic_pipeline):
    """Test running entire pipeline"""
    start_time = time.perf_counter()
    basic_pipeline.run()
    duration = time.perf_counter() - start_time
    assert duration >= 0.3 and duration <= 0.4  # 3 tasks * 0.1s each


def test_run_step(basic_pipeline):
    """Test running specific step"""
    start_time = time.perf_counter()
    basic_pipeline.run_step(0)
    duration = time.perf_counter() - start_time
    assert duration >= 0.1  # 1 task * 0.1s


def test_run_from_step(basic_pipeline):
    """Test running pipeline from specific step"""
    start_time = time.perf_counter()
    basic_pipeline.run_from_step(1)
    duration = time.perf_counter() - start_time
    assert duration >= 0.2 and duration <= 0.3  # 2 tasks * 0.1s each


def test_run_until_step(basic_pipeline):
    """Test running pipeline until specific step"""
    start_time = time.perf_counter()
    basic_pipeline.run_until_step(2)
    duration = time.perf_counter() - start_time
    assert duration >= 0.2  # 2 tasks * 0.1s each


def test_invalid_step_number(basic_pipeline):
    """Test error handling for invalid step numbers"""
    with pytest.raises(IndexError):
        basic_pipeline.run_step(10)
    with pytest.raises(IndexError):
        basic_pipeline.run_from_step(10)
    with pytest.raises(IndexError):
        basic_pipeline.run_until_step(10)


def test_pipeline_with_mixed_steps(empty_pipeline, parallel_block):
    """Test pipeline with different types of steps"""
    empty_pipeline.add_step(DelayedTask(slow_task, 1))
    parallel_block.add_step(DelayedTask(slow_task, 2))
    empty_pipeline.add_step(parallel_block)

    start_time = time.perf_counter()
    empty_pipeline.run()
    duration = time.perf_counter() - start_time
    assert duration >= 0.2 and duration < 0.3


def test_pipeline_repr_str(basic_pipeline):
    """Test string representations"""
    assert "Pipeline" in str(basic_pipeline)
    assert "Pipeline" in repr(basic_pipeline)
    assert "steps" in repr(basic_pipeline)


def test_pipeline_getitem(basic_pipeline):
    """Test indexing access"""
    assert isinstance(basic_pipeline[0], DelayedTask)
    len(basic_pipeline)  # Ensure no errors
    with pytest.raises(IndexError):
        _ = basic_pipeline[10]


def test_invalid_step_type():
    """Test adding invalid step type"""
    pipe = Pipeline()
    with pytest.raises(TypeError):
        pipe.add_step("not a valid step")


if __name__ == "__main__":
    pytest.main()
