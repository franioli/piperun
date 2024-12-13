import contextlib
import io
import logging
import subprocess
import sys
import time
from typing import List

logger = logging.getLogger("pyasp")


def cmd_list_to_string(cmd_list: List[str]) -> str:
    """
    Convert a list of command arguments to a single string.

    Args:
        cmd_list (List[str]): The list of command arguments.

    Returns:
        str: The command as a single string.
    """
    if not cmd_list:
        raise ValueError("cmd_list must not be empty")
    elif not isinstance(cmd_list, list):
        raise TypeError(
            "cmd_list must be a list of strings (numbers are automatically casted to strings)"
        )
    try:
        cmd_list = [str(arg) for arg in cmd_list]
    except TypeError as e:
        logger.error(f"Failed to convert command list to string: {e}")
        raise

    return " ".join(cmd_list)


def cmd_string_to_list(cmd_str: str) -> List[str]:
    """
    Convert a command string to a list of arguments.

    Args:
        cmd_str (str): The command as a single string.

    Returns:
        List[str]: The list of command arguments.
    """
    if not isinstance(cmd_str, str):
        raise TypeError("cmd_str must be a string")
    return cmd_str.split()


class OutputCapture:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                logger.error("Failed with output:\n%s", self.out.getvalue())
        sys.stdout.flush()


def run_command(command: List[str] | str, silent: bool = False) -> bool:
    """
    Run a shell command, capture output in real time, and handle errors.

    Args:
        command (List[str] | str): Command and arguments to execute.
        silent (bool): Suppress output.
        kwargs: Additional keyword arguments to pass to subprocess.run.

    Returns:
        CompletedProcess: The command output.
    """
    if isinstance(command, str):
        command = cmd_string_to_list(command)
    elif isinstance(command, list):
        pass
    else:
        raise TypeError("command must be a list of strings or a single string")

    if silent:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True

    logger.debug(f"Executing command: {command}")

    with OutputCapture(verbose=True):
        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
            )
            logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            try:
                error_message = (
                    e.stdout if e.stdout and "ERROR" in e.stdout else e.stderr
                )
                if error_message:
                    error_index = error_message.find("ERROR")
                    if error_index != -1:
                        error_message = error_message[error_index:]
            except Exception:
                error_message = f"Command {command[0]} failed with error code {e.returncode}, but the error message could not be retrieved."

            raise RuntimeError(
                f"Command {command[0]} failed with error code {e.returncode}: {error_message}"
            )

        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f"Function {command[0]} took {total_time:.4f} s.")

    return True


class Command:
    cmd = []
    name = ""
    _elaspsed_time = None
    silent = False

    """
    A class to represent a command with its parameters for execution.

    Attributes:
        cmd (list): The command and its parameters.
        name (str): A name for the command, default is "Command".
        silent (bool): If True, suppress output during execution.
        kwargs (dict): Additional keyword arguments for command execution.
    """

    def __init__(
        self,
        cmd: str | list[str],
        name: str = "Command",
        silent: bool = False,
        **kwargs,
    ):
        """
        Initializes a Command instance.

        Args:
            cmd (str | list[str]): The base command as a string or a list of strings.
            name (str, optional): The name of the command. Default is "Command".
            silent (bool, optional): Suppress output if True. Default is False.
            **kwargs: Additional keyword arguments for command execution.

        Raises:
            TypeError: If `cmd` is neither a string nor a list of strings.
        """
        if isinstance(cmd, str):
            cmd = cmd_string_to_list(cmd)
        elif isinstance(cmd, list):
            pass
        else:
            raise TypeError("cmd must be a list of strings or a single string")
        self.cmd = cmd
        self.name = name
        self.silent = silent
        self.kwargs = kwargs

    def __str__(self) -> str:
        """
        Returns a string representation of the Command.

        Returns:
            str: The string representation of the Command.
        """
        return f"{cmd_list_to_string(self.cmd)}"

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the Command for debugging.

        Returns:
            str: The string representation for debugging.
        """
        return f"{self.__class__.__name__}({self.name}: {self})"

    def __call__(self) -> bool:
        """
        Executes the command by calling the run_command function with the command parameters.
        """
        return self.run()

    @property
    def elapsed_time(self) -> float:
        """
        Return the elapsed time of the last command execution.

        Returns:
            float: The elapsed time in seconds.
        """
        if self._elaspsed_time is None:
            logger.info("The step has not been executed yet.")
        return self._elaspsed_time

    def extend(self, *args, **kwargs):
        """
        Extends the command with additional arguments.

        Args:
            *args: Positional arguments to extend the command.
            **kwargs: Keyword arguments to add as parameters to the command.
        """
        if not args and not kwargs:
            return

        # Extend the command with the positional arguments
        for arg in args:
            # Handle the case where the argument is a list: Convert list elements to strings
            if isinstance(arg, list):
                self.cmd.extend(map(str, arg))
                continue

            # Convert single elements to string and append to the command
            self.cmd.append(str(arg))

        # Extend the command with additional keyword arguments
        for key, value in kwargs.items():
            # Handle boolean flags (no value)
            if isinstance(value, bool) and value is True:
                self.cmd.append(f"{key}")
                continue
            if isinstance(value, bool) and value is False:
                continue

            # If a key has an empty string value, add the key without a value
            if isinstance(value, str) and value == "":
                self.cmd.append(f"{key}")
                continue

            # Handle no value for the key
            if value is None:
                continue

            # Handle the case where the value is a list or tuple
            if isinstance(value, list) or isinstance(value, tuple):
                self.cmd.extend([f"{key}", *map(str, value)])
                continue

            # Add the key and value as separate arguments
            # If the key include '--' or '-', directly add it
            if key.startswith("--") or key.startswith("-"):
                self.cmd.extend([f"{key}", str(value)])
            # Otherwise, transform the key to a command line argument format
            else:
                # Replace underscores with hyphens
                key = key.replace("_", "-")

                if len(key) == 1:  # Short keys
                    if isinstance(value, bool):
                        if value:
                            self.cmd.extend([f"-{key}"])
                    else:
                        self.cmd.extend([f"-{key}", str(value)])

                elif len(key) > 1:  # Long keys
                    if isinstance(value, bool):
                        if value:
                            self.cmd.extend([f"--{key}"])
                    else:
                        self.cmd.extend([f"--{key}", str(value)])

                else:
                    raise ValueError(f"Invalid key: {key}")

    def run(self):
        """
        Runs the command using the run_command function.

        Additional arguments from kwargs are passed to the run_command function.
        """
        start_time = time.perf_counter()
        ret = run_command(self.cmd, silent=self.silent)
        self._elaspsed_time = time.perf_counter() - start_time

        logger.info(f"Command {self.name} took {self._elaspsed_time:.4f}s.")

        return ret

    def get_parameters(self) -> list[str]:
        """
        Returns a list of command parameters and their values (if any).

        Parameters are considered those that start with '--' or '-'.

        Returns:
            list[str]: A list of parameters and their associated values.
        """
        parameters = []
        skip_next = False

        for i, arg in enumerate(self.cmd):
            if skip_next:
                skip_next = False
                continue

            # Check if the argument is a parameter (starts with -- or -)
            if arg.startswith("--") or arg.startswith("-"):
                parameters.append(arg)
                # If the next argument is not a parameter, it is likely a value for this parameter
                if (
                    i + 1 < len(self.cmd)
                    and not self.cmd[i + 1].startswith("--")
                    and not self.cmd[i + 1].startswith("-")
                ):
                    parameters.append(self.cmd[i + 1])
                    skip_next = True  # Skip the next argument since it's a value

        return parameters

    def get_positional_arguments(self) -> List[str]:
        """
        Returns a list of positional arguments.

        Positional arguments are considered those that do not start with '--' or '-'.

        Returns:
            list[str]: A list of positional arguments.
        """
        positional_args = []
        skip_next = False

        # Start from index 1 to skip the command name
        for i, arg in enumerate(self.cmd[1:], start=1):
            if skip_next:
                skip_next = False
                continue

            if not arg.startswith("--") and not arg.startswith("-"):
                positional_args.append(arg)
            elif (
                arg.startswith("-")
                and i + 1 < len(self.cmd)
                and not self.cmd[i + 1].startswith("-")
            ):
                skip_next = True

        return positional_args

    def get_keywork_argument(self, key: str) -> str:
        """
        Returns the value of a keyword argument.

        Args:
            key (str): The keyword argument to retrieve.

        Returns:
            str: The value of the keyword argument.
        """
        for i, arg in enumerate(self.cmd):
            if arg.startswith("--") or arg.startswith("-"):
                if arg == f"--{key}" or arg == f"-{key}":
                    return self.cmd[i + 1]
            elif arg == key:
                return self.cmd[i + 1]
        return None


if __name__ == "__main__":
    # Test cmd_list_to_string
    cmd = "ls -l ."
    print(cmd_string_to_list(cmd))

    # Test initialization of Command and run method
    cmd = Command("python --version")
    print(cmd)
    print(cmd)
    cmd()
    cmd.run()

    # Add positional and keyword arguments
    cmd = Command("ls")
    positional_args = ["image.tif", "metadata.xml"]
    keyword_args = {"t": "rpc", "e": "rpc"}
    cmd.extend(positional_args, **keyword_args)
    print(cmd)

    # Get parameters and positional arguments
    print(cmd.get_positional_arguments())
    print(cmd.get_parameters())

    # Add keyword arguments with no value
    cmd.extend(a=True, b="")
    print(cmd)

    # Add keyword arguments with long keys and values
    cmd = Command("ls")
    cmd.extend(abcd="efgh")
    print(cmd)

    cmd = Command("ls")
    cmd.extend(**{"--abcd": "efgh"})
    print(cmd)

    # Add positional arguments with list values
    cmd = Command("ls")
    cmd.extend([1, 2, 3])
    print(cmd)

    # Add positional arguments with values in tuple
    cmd.extend(("a", "b", "c"))
    print(cmd)  # NOTE: WRONG BEHAVIOR!!FIX IT.

    # Add keyword arguments with dictionary
    cmd = Command("ls")
    cmd.extend(a=[1, 2, 3])
    print(cmd)  # NOTE: WRONG BEHAVIOR!! THEY SHOULD BE KEYWORDS ARGUMENTS! FIX IT

    cmd = Command("ls")
    cmd.extend(**{"a": [1, 2, 3], "b": ("x", "y", "z")})
    print(cmd)  # NOTE: WRONG BEHAVIOR!! THEY SHOULD BE KEYWORDS ARGUMENTS! FIX IT
