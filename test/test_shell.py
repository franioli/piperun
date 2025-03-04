import subprocess
import sys

import pytest

from piperun.shell import (
    Command,
    OutputCapture,
    cmd_list_to_string,
    cmd_string_to_list,
    run_command,
)


def test_cmd_list_to_string():
    assert cmd_list_to_string(["ls", "-l"]) == "ls -l"
    assert cmd_list_to_string(["echo", "hello", "world"]) == "echo hello world"
    with pytest.raises(ValueError):
        cmd_list_to_string([])
    with pytest.raises(TypeError):
        cmd_list_to_string("not a list")


def test_cmd_string_to_list():
    assert cmd_string_to_list("ls -l") == ["ls", "-l"]
    assert cmd_string_to_list("echo 'Hello, World!'") == ["echo", "'Hello,", "World!'"]
    with pytest.raises(TypeError):
        cmd_string_to_list(123)


def test_run_command(mocker):
    mocker.patch("subprocess.run")
    subprocess.run.return_value = subprocess.CompletedProcess(
        args=["ls"], returncode=0, stdout="output", stderr=""
    )

    result = run_command(["ls"])
    assert result

    result = run_command("ls", silent=True)
    assert result

    with pytest.raises(TypeError):
        run_command(123)


def test_output_capture(mocker):
    mocker.patch("sys.stdout.flush")
    with OutputCapture(verbose=False):
        pass
    sys.stdout.flush.assert_called_once()


def test_command_initialization():
    cmd = Command("python --version")
    assert str(cmd) == "python --version"
    assert cmd.cmd == ["python", "--version"]


def test_command_run_method(mocker):
    mocker.patch("subprocess.run")
    subprocess.run.return_value = subprocess.CompletedProcess(
        args=["ls"], returncode=0, stdout="output", stderr=""
    )
    cmd = Command("ls -l")
    assert cmd()
    assert cmd.run()


def test_command_get_positional_arguments():
    cmd = Command("ls image.tif metadata.xml -t rpc -e rpc")
    assert cmd.get_positional_arguments() == ["image.tif", "metadata.xml"]


def test_command_get_positional_arguments_after_extend():
    cmd = Command("cat")
    cmd.extend("file.txt", t="rpc", longkey="value")
    assert cmd.get_positional_arguments() == ["file.txt"]


def test_command_get_parameters():
    cmd = Command("ls image.tif metadata.xml -t rpc -e rpc")
    assert cmd.get_parameters() == ["-t", "rpc", "-e", "rpc"]


def test_command_get_single_keyword_argument_after_extend():
    cmd = Command("cat")
    cmd.extend("file.txt", t="rpc", longkey="value", key_with_underscore="value")
    assert cmd.get_keywork_argument("t") == "rpc"


def test_command_extend_positional_and_keyword_arguments():
    cmd = Command("ls")
    positional_args = ["image.tif", "metadata.xml"]
    keyword_args = {"t": "rpc", "e": "rpc"}
    cmd.extend(positional_args, **keyword_args)
    assert str(cmd) == "ls image.tif metadata.xml -t rpc -e rpc"


def test_command_extend_with_positional_and_keyword_arguments():
    # NOTE: if a keyword argument with underscore is passed at the Command  onstructor, underscore will NOT be replaced by hyphen
    cmd = Command("cat")
    cmd.extend("file.txt", t="rpc", longkey="value", key_with_underscore="value")
    assert cmd.cmd == cmd_string_to_list(
        "cat file.txt -t rpc --longkey value --key_with_underscore value"
    )


# TODO: decide if we should replace underscore with hyphen in keyword arguments
# def test_command_get_parameters_after_extend():
#     cmd = Command("cat")
#     cmd.extend("file.txt", t="rpc", longkey="value", key_with_underscore="value")
#     assert cmd.get_parameters() == [
#         "-t",
#         "rpc",
#         "--longkey",
#         "value",
#         "--key-with-underscore",
#         "value",
#     ]


def test_command_extend_keyword_arguments_with_true_value():
    cmd = Command("ls")
    cmd.extend(a=True)
    assert str(cmd) == "ls -a"


# TODO: Fix Command class to pass this test
def test_command_extend_keyword_arguments_with_false_value():
    cmd = Command("ls")
    cmd.extend(a=False)
    assert str(cmd) == "ls -a false"


def test_command_extend_keyword_arguments_with_empty_string():
    cmd = Command("ls")
    cmd.extend(b="")
    assert str(cmd) == "ls -b"


def test_command_extend_keyword_arguments_with_long_keys_and_values():
    cmd = Command("ls")
    cmd.extend(abcd="efgh")
    assert str(cmd) == "ls --abcd efgh"


def test_command_extend_keyword_arguments_with_long_keys_with_double_dash():
    cmd = Command("ls")
    cmd.extend(**{"--abcd": "efgh"})
    assert str(cmd) == "ls --abcd efgh"


def test_command_extend_positional_arguments_with_list_values():
    cmd = Command("ls")
    cmd.extend([1, 2, 3])
    assert str(cmd) == "ls 1 2 3"


def test_command_extend_keyword_arguments_with_list_values():
    cmd = Command("ls")
    cmd.extend(a=[1, 2, 3])
    assert str(cmd) == "ls -a 1 2 3"
    assert str(cmd) != "ls -a 1 -a 2 -a 3"


def test_command_extend_positional_arguments_with_tuple_values():
    cmd = Command("ls")
    cmd.extend(("a", "b", "c"))
    assert str(cmd) == "ls a b c"


def test_command_extend_keyword_arguments_with_tuple_values_in_dict():
    cmd = Command("ls")
    cmd.extend(**{"a": [1, 2, 3], "b": ("x", "y", "z")})
    assert str(cmd) == "ls -a 1 2 3 -b x y z"


if __name__ == "__main__":
    pytest.main()
