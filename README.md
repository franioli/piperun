# Piperun - Command Pipeline Framework for Python

piperun is a flexible framework for building and executing command pipelines in Python. It provides tools for running shell commands, creating task pipelines, and executing operations in parallel.

## Overview

piperun simplifies the process of building complex command workflows by providing:

- Command construction and execution with detailed control
- Sequential pipeline execution
- Parallel processing capabilities
- Integration with Dask for distributed computation

piperun was designed with the following principles in mind:

1. **Simplicity**: Easy to use with intuitive interfaces
2. **Flexibility**: Works with any callable or command
3. **Composability**: Build complex workflows from simple pieces
4. **Performance**: Efficient execution with parallel processing capabilities
5. **Control**: Detailed control over execution flow

## Installation

Install piperun using pip (not yet available):
```bash
pip install piperun
```

Install from source in editable mode:
```bash
git clone https://github.com/franioli/piperun.git
cd piperun
pip install -e .
```

## Core Components

### Command

The `Command` class provides an intuitive interface for constructing and executing shell commands:

```python
from piperun import Command

# Create a simple command
cmd = Command("ls -l")
cmd.run()

# Add arguments dynamically
cmd = Command("parallel_stereo")
cmd.extend("image1.tif", "image2.tif", t="rpc", max_level=2)
cmd.run()
```

Key features:
- Handles both positional and keyword arguments
- Converts Python arguments to command-line format
- Provides execution timing
- Captures command output
- Supports boolean flags and various parameter formats

### Pipeline

The `Pipeline` class enables chaining multiple processing steps:

```python
from piperun import Pipeline, Command

# Create a pipeline
pipeline = Pipeline()

# Add steps
pipeline.add_step(Command("mkdir -p output"))
pipeline.add_step(Command("convert input.jpg output/output.png"))

# Execute all steps
pipeline.run()
```

Features:
- Sequential execution of steps
- Control for running specific steps or ranges
- Support for any step with a `run()` method
- Nested pipeline capability

### DelayedTask

The `DelayedTask` class integrates with Dask for delayed execution:

```python
from piperun import DelayedTask

# Create a delayed task
def process_data(x):
    return x * 2

task = DelayedTask(process_data, 10)
result = task.compute()  # Executes when needed
```

Features:
- Lazy evaluation of tasks
- Execution timing measurement
- Visualization of task graphs

### ParallelBlock

The `ParallelBlock` class enables concurrent execution of multiple steps:

```python
from piperun import ParallelBlock, Command

# Create commands
commands = [
    Command(f"process_file {i}.txt") 
    for i in range(10)
]

# Run in parallel
with ParallelBlock(commands, workers=4) as block:
    block.run()
```

Features:
- Automatic Dask cluster management
- Configurable worker count
- Support for both Command and DelayedTask objects



## Advanced Usage

### Pipeline Composition

Pipelines can be composed of various step types, including other pipelines:

```python
# Create nested pipelines
preprocessing = Pipeline([
    Command("clean_data input.csv"),
    Command("validate_data input.csv")
])

processing = Pipeline([
    Command("process_data input.csv output.csv")
])

# Combine pipelines
main_pipeline = Pipeline()
main_pipeline.add_step(preprocessing)
main_pipeline.add_step(processing)
main_pipeline.run()
```

### Parallel Execution with Custom Worker Count

```python
# Run operations in parallel with custom worker count
parallel_tasks = ParallelBlock(workers=8)

for file in input_files:
    parallel_tasks.add_step(Command(f"process {file}"))

parallel_tasks.run()
```

### Flow Control in Pipelines

```python
# Run specific pipeline segments
pipeline = Pipeline([...])

# Run only step 3
pipeline.run_step(3)

# Run from step 2 to the end
pipeline.run_from_step(2)

# Run up to step 4 (not including step 4)
pipeline.run_until_step(4)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.