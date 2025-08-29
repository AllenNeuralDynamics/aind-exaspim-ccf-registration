"""
Utility functions
"""

import json
import logging
import multiprocessing
import os
import platform
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import boto3
from urllib.parse import urlparse
import psutil


def create_folder(dest_dir: str, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: str
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def get_gpu_metrics():
    """
    Retrieves the GPU metrics in current time.

    Returns
    -------
    Dict
        Dictionary that contains the gpu index,
        utilization and memory at current time.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip().split("\n")
    gpu_metrics = {}
    for line in output:
        gpu_index, gpu_util, mem_util = map(float, line.split(","))

        gpu_metrics[int(gpu_index)] = {
            "gpu_utilization": float(gpu_util),
            "memory_utilization": float(mem_util),
        }

    return gpu_metrics


def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
    gpu_resources: Optional[Dict] = None,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    gpu_resources: Dict
        Dict to save the gpu resources

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        # GPU resources
        if gpu_resources is not None:
            gpu_metrics = get_gpu_metrics()

            for curr_idx, vals in gpu_metrics.items():
                gpu_resources[curr_idx]["gpu_utilization"].append(vals["gpu_utilization"])
                gpu_resources[curr_idx]["memory_utilization"].append(vals["memory_utilization"])

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
    gpu_resources: Optional[List] = {},
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)
    gpu_len = len(gpu_resources.keys())

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_points[:min_len], memory_usages[:min_len], label="Memory Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight")

    if gpu_len:
        gpu_indexes = list(gpu_resources.keys())

        plt.figure(figsize=(15, 15))

        for idx, gpu_idx in enumerate(gpu_indexes):
            plt.subplot(2, 1, 1)
            plt.plot(
                time_points[:min_len],
                gpu_resources[gpu_idx]["gpu_utilization"][:min_len],
                label="GPU Usage",
            )
            plt.xlabel("Time (s)")
            plt.ylabel("GPU Usage (%)")
            plt.title(f"GPU {gpu_idx} - Usage Over Time")
            plt.grid(True)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(
                time_points[:min_len],
                gpu_resources[gpu_idx]["memory_utilization"][:min_len],
                label="Memory Usage",
            )
            plt.xlabel("Time (s)")
            plt.ylabel("GPU Memory Usage (%)")
            plt.title(f"GPU {gpu_idx} - Memory Usage Over Time")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(
                f"{output_path}/{prefix}_GPU_{gpu_idx}_compute_resources.png",
                bbox_inches="tight",
            )


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def create_logger(output_log_path: str, mode: Optional[str] = "w") -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    mode: Optional[str]
        Open mode.
        Default: 'a'

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    # CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/segmentation_log.log"  # _{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, mode),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


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
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def print_system_information(logger: logging.Logger, code_ocean: Optional[bool] = False):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object

    code_ocean: Optional[bool]
        If the instance is within a Code Ocean environment.
    """
    sep = "=" * 20

    if code_ocean:
        co_memory = int(os.environ.get("CO_MEMORY"))
        # System info
        logger.info(f"{sep} Code Ocean Information {sep}")
        logger.info(f"Code Ocean assigned cores: {get_code_ocean_cpu_limit()}")
        logger.info(f"Code Ocean assigned memory: {get_size(co_memory)}")
        logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
        logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
        logger.info(f"Is pipeline execution?: {os.environ.get('AWS_BATCH_JOB_ID')}")

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
    logger.info(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

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
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
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


def pad_array_n_d(arr: np.ndarray, dim: int = 5) -> np.ndarray:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def parse_zarr_metadata(metadata: Dict, multiscale: Optional[str] = None) -> Dict:
    """
    Parses the zarr metadata and retrieves
    the metadata we need in the correct format.

    Parameters
    ----------
    metadata: Dict
        Metadata dictionary that contains ".zattrs" and
        ".zarray"

    multiscale: Optional[str]
        Multiscale we're retieving the metadata for.
        Default: None

    Returns
    -------
    Dict
        Dictionary with the metadata we need
    """
    parsed_metadata = {"axes": {}}
    zattrs = metadata.get(".zattrs")
    # zarray = metadata.get('.zarray')

    if zattrs is not None:
        # SmartSPIM
        multiscales = zattrs.get("multiscales")[0]
        axes = multiscales.get("axes")
        datasets = multiscales.get("datasets")

        dataset_res = None

        for d in datasets:
            if d["path"] == multiscale:
                dataset_res = d["coordinateTransformations"][0]["scale"]
                break

        for idx in range(len(axes)):
            ax = axes[idx]
            parsed_metadata["axes"][ax["name"]] = {
                "unit": ax.get("unit"),
                "type": ax.get("type"),
                "scale": dataset_res[idx],
            }

    else:
        # Exaspim
        multiscales = metadata.get("multiscales")[0]
        axes = multiscales.get("axes")
        datasets = multiscales.get("datasets")

#         print(f"\n*multiscales*: {multiscales}")
#         print(f"\n*axes*: {axes}")
#         print(f"\n*datasets*: {datasets}")
        
        dataset_res = None
        for d in datasets:
            if d["path"] == multiscale:
                dataset_res = d["coordinateTransformations"][0]["scale"]
                break
                
        # print(f"*dataset_res*: {dataset_res}")
        for idx in range(len(axes)):
            ax = axes[idx]
            parsed_metadata["axes"][ax["name"]] = {
                "unit": ax.get("unit"),
                "type": ax.get("type"),
                "scale": dataset_res[idx],
                # "scale": dataset_res,
            }
        # print(f"\n*parsed_metadata*: {parsed_metadata}")

    return parsed_metadata


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary

def load_json(data_path: str, keyname: Optional[str] = ".zattrs") -> dict:
    """
    Loads a JSON file from either a local path or S3.

    Parameters
    ----------
    data_path: str
        Path to the data. Can be:
        - Local file path (e.g., "/path/to/file.json" or "/path/to/zarr/dataset")
        - S3 path (e.g., "s3://bucket-name/path/to/zarr/dataset")
    keyname: Optional[str]
        Name of the JSON file to load. Default is ".zattrs".
        - For local paths: this is appended to data_path
        - For S3 paths: this is appended to the S3 key

    Returns
    -------
    dict
        Dictionary with the JSON data

    Raises
    ------
    FileNotFoundError
        If the local file doesn't exist
    Exception
        If there's an error accessing S3 or parsing JSON
    """
    
    # Check if it's an S3 path
    if data_path.startswith('s3://'):
        return _load_json_from_s3(data_path, keyname)
    else:
        return _load_json_from_local(data_path, keyname)


def _load_json_from_s3(s3_data_path: str, keyname: str) -> dict:
    """
    Load JSON from S3.
    """
    parsed = urlparse(s3_data_path)
    bucket_name = parsed.netloc
    key = f"{parsed.path.lstrip('/')}/{keyname}"
    
    try:
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data
    except Exception as e:
        raise Exception(f"Error loading JSON from S3 path '{s3_data_path}/{keyname}': {str(e)}")


def _load_json_from_local(local_path: str, keyname: str) -> dict:
    """
    Load JSON from local file system.
    """
    # Construct the full file path
    if keyname:
        full_path = os.path.join(local_path, keyname)
    else:
        full_path = local_path
    
    # Handle case where the path already includes the filename
    if not os.path.exists(full_path) and os.path.exists(local_path):
        # If the constructed path doesn't exist but the original path does,
        # assume the original path is the complete file path
        full_path = local_path
    
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at path: {full_path}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing JSON file '{full_path}': {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading JSON file '{full_path}': {str(e)}")
