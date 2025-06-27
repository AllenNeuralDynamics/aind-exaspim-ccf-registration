"""
Utility functions for CCF registration pipeline.

This module contains utility functions for logging, file operations, 
data processing tracking, and system information.
"""

import json
import logging
import multiprocessing
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
import re
import s3fs

import psutil
import pydantic
from aind_exaspim_ccf_reg.configs import PathLike
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing)

def extract_dataset_id(s3_path: str) -> Optional[str]:
    """
    Extract dataset ID from S3 path.
    
    Parameters
    ----------
    s3_path : str
        S3 path containing dataset information
        
    Returns
    -------
    Optional[str]
        Extracted dataset ID or None if not found
    """
    match = re.search(r"exaSPIM_(\d{6})_", s3_path)
    return match.group(1) if match else None


def get_available_memory() -> float:
    """
    Returns the available memory in GBs.

    Returns
    -------
    float
        Available memory in GBs
    """
    available_memory_bytes = os.environ.get("CO_MEMORY", None)

    if available_memory_bytes is not None:
        available_memory_bytes = int(available_memory_bytes)
    else:
        # Use psutil to get available virtual memory in bytes
        available_memory_bytes = psutil.virtual_memory().available

    available_memory_gb = available_memory_bytes / (1024**3)
    return available_memory_gb


def create_logger(output_log_path: PathLike) -> logging.Logger:
    """
    Creates a logger that generates output logs to a specific path.

    Parameters
    ----------
    output_log_path : PathLike
        Path where the log is going to be stored

    Returns
    -------
    logging.Logger
        Created logger pointing to the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    LOGS_FILE = f"{output_log_path}/register_process.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Execution datetime: {CURR_DATE_TIME}")

    return logger


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ----------
    filepath : str
        Path where the json is located.

    Returns
    -------
    dict
        Dictionary with the data the json has.
    """
    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def parse_s3_path(dataset_path: str) -> str:
    """
    Parse S3 path to extract dataset information.
    """
    m = re.search("(s3:\/\/.+?\/.+?_.+?_(.+?_.+?))_.*", dataset_path)
    return m.group(1)


def download_file(s3path: str, lpath: str) -> None:
    """
    Download file from S3 to local path.
    
    Parameters
    ----------
    s3path : str
        S3 path to download from
    lpath : str
        Local path to download to
    """
    fs = s3fs.S3FileSystem(anon=True)
    fs.download(s3path, lpath)


def prepare_config_sample(
    dataset_path: PathLike,
    logger: logging.Logger,
    dataset_id: str,
    acquisition_output: str,
) -> str:
    """
    Prepare configuration sample by downloading acquisition metadata.
    
    Parameters
    ----------
    dataset_path : PathLike
        Path to the dataset
    logger : logging.Logger
        Logger instance for output messages
    dataset_id : str
        Dataset identifier
    acquisition_output : str
        Output path for acquisition metadata
        
    Returns
    -------
    str
        Path to the downloaded acquisition metadata file
    """
    dataset_name = parse_s3_path(dataset_path=dataset_path)
    acquisition_path = f"{dataset_name}/acquisition.json"
    if "s3://aind-scratch-data/" in acquisition_path:
        acquisition_path = acquisition_path.replace("s3://aind-scratch-data/", "s3://aind-open-data/")
    logger.info(f"Acquisition path: {acquisition_path}")

    acquisition_output = f"{acquisition_output}acquisition_{dataset_id}.json"

    try:
        download_file(acquisition_path, acquisition_output)
    except FileNotFoundError:
        print(f"Error: {acquisition_path} not found.")
        return

    return acquisition_output


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ----------
    dest_dir : PathLike
        Path where the folder will be created if it does not exist.
    verbose : Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------
    OSError
        If the folder cannot be created.
    """
    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: PathLike,
    processor_full_name: str,
    pipeline_version: str,
) -> None:
    """
    Generate processing metadata and save to file.

    Parameters
    ----------
    data_processes : List[DataProcess]
        List of data processing steps
    dest_processing : PathLike
        Destination path for processing metadata
    processor_full_name : str
        Full name of the processor
    pipeline_version : str
        Version of the pipeline
    """
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://codeocean.allenneuraldynamics.org/capsule/1087961/tree",
        note="Metadata for the CCF Atlas Registration step",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata of ccf registration \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(output_directory=dest_processing)


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        cfs_period_us = int(fp.read())
    container_cpus = cfs_quota_us // cfs_period_us
    # For physical machine, the `cfs_quota_us` could be '-1'
    return (
        psutil.cpu_count(logical=False)
        if container_cpus < 1
        else container_cpus
    )


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    co_memory = int(os.environ.get("CO_MEMORY"))
    # System info
    sep = "=" * 40
    logger.info(f"{sep} Code Ocean Information {sep}")
    logger.info(f"Code Ocean assigned cores: {get_code_ocean_cpu_limit()}")
    logger.info(f"Code Ocean assigned memory: {get_size(co_memory)}")
    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(
        f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}"
    )

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(
        f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}"
    )

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(
        psutil.cpu_percent(percpu=True, interval=1)
    ):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


def save_string_to_txt(txt: str, filepath: str, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------
    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")
