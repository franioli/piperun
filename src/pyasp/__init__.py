import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Import submodules
from . import asp, preproc, spot5, thirdparty, utils

# Import classes and functions
from .path_manager import PathManager
from .pipeline import DelayedTask, ParallelBlock, Pipeline
from .shell import Command
from .utils.timer import Timer

__version__ = "0.0.4"


class PyASPError(Exception):
    """Custom exception for Ames Stereo Pipeline errors."""

    pass


def setup_logger(
    level: str | int = logging.INFO,
    name="pyasp",
    log_to_file: bool = True,
    log_folder: Path | str = "./.logs",
):
    """
    Reconfigures the 'pyasp' logger with new parameters by calling setup_logger.

    Args:
        level (str | int): The logging level (e.g., 'info', 'debug', 'warning').
        name (str): The name of the logger.
        log_to_file (bool, optional): Whether to log to a file. Defaults to True.
        log_folder (Path, optional): Path to the directory for the log file if log_to_file is True. Defaults to "./.logs".

    Returns:
        logging.Logger: The reconfigured 'pyasp' logger.

    Example:
        >>> import logging
        >>> from pyasp import setup_logger
        >>> logger = setup_logger(level=logging.DEBUG, log_to_file=False)
        >>> logger.debug("This is a debug message")
    """
    return utils.logger.setup_logger(
        level=level,
        name=name,
        log_to_file=log_to_file,
        log_folder=log_folder,
    )


# Setup logger and timer for the package
logger = setup_logger(
    level=logging.INFO,
    name="pyasp",
    log_to_file=True,
    log_folder="./.logs",
)
timer = Timer(logger=logger)


def check_asp_binary():
    """
    Check if the Ames Stereo Pipeline binaries are in the PATH.

    Returns:
        bool: True if the ASP binaries are in the PATH, False otherwise.
    """
    return shutil.which("parallel_stereo") is not None


def add_asp_binary(path: Path) -> bool:
    """
    Add the Ames Stereo Pipeline binaries to the PATH.

    Args:
        path (Path): The path to the ASP binaries.

    Returns:
        bool: True if the ASP binaries were added to the PATH, False otherwise.
    """
    if not isinstance(path, str | Path):
        raise TypeError("directory must be a string or Path object")
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")

    os.environ.update({"PATH": f"{str(path)}:{os.environ['PATH']}"})

    return check_asp_binary()


def get_latest_asp_release():
    """Get latest ASP release URL from GitHub."""
    url = "https://api.github.com/repos/NeoGeographyToolkit/StereoPipeline/releases/latest"
    response = requests.get(url)
    if response.status_code != 200:
        raise PyASPError("Failed to fetch latest ASP release info")

    assets = response.json()["assets"]
    for asset in assets:
        if "x86_64-Linux.tar.bz2" in asset["name"]:
            return asset["browser_download_url"]
    raise PyASPError("No Linux binary found in latest release")


def download_asp_binaries():
    """Download and setup ASP binaries."""
    try:
        download_url = get_latest_asp_release()
        logger.info(f"Downloading ASP binaries from {download_url}...")

        # Create a temporary directory that persists until we're done
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "asp.tar.bz2"

            # Download with progress bar
            response = requests.get(download_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with logging_redirect_tqdm([logger]):
                with open(tmp_path, "wb") as f:
                    with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
                        for data in response.iter_content(chunk_size=8192):
                            size = f.write(data)
                        pbar.update(size)

            # Verify the downloaded file exists and has content
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise PyASPError("Downloaded file is empty or missing")

            logger.info("Download complete. Extracting files...")

            # Extract to home directory
            home = Path.home()
            try:
                with tarfile.open(tmp_path, "r:bz2") as tar:
                    tar.extractall(path=home)
            except tarfile.ReadError as e:
                raise PyASPError(f"Failed to extract archive: {e}") from e

            # Find extracted directory
            asp_dirs = list(home.glob("StereoPipeline*"))
            if not asp_dirs:
                raise PyASPError("Failed to extract ASP binaries")

            bin_dir = asp_dirs[0] / "bin"
            if not bin_dir.exists():
                raise PyASPError("Binary directory not found in extracted files")

            return bin_dir

    except requests.RequestException as e:
        raise PyASPError(f"Download failed: {e}") from e
    except Exception as e:
        raise PyASPError(f"Failed to download ASP binaries: {e}") from e


# Check if ASP binaries exist in PATH, otherwise download and add them
if not check_asp_binary():
    try:
        # Try to find ASP binaries in home directory
        bin_paths = list(Path.home().glob("StereoPipeline*/bin"))
        ASP_PATH = bin_paths[0] if bin_paths else None

        if not ASP_PATH:
            logger.info("ASP binaries not found. Downloading latest release...")
            ASP_PATH = download_asp_binaries()

        if ASP_PATH.exists():
            os.environ.update({"PATH": f"{str(ASP_PATH)}:{os.environ['PATH']}"})
            if check_asp_binary():
                logger.info(f"ASP binaries added to PATH: {ASP_PATH}")
            else:
                raise PyASPError("Failed to add ASP binaries to PATH")
        else:
            raise PyASPError("AmesStereoPipeline binaries not found. ")
    except PyASPError as e:
        logger.warning(
            f"{e}. Please add them manually with 'pyasp.add_asp_binary(`Path/to/asp/binaries`)'."
        )
