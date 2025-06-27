"""
Preprocess exaspim data
Reference: https://github.com/AllenNeuralDynamics/aind-axis-utils/blob/feat-adjust-axes/src/aind_axis_utils/axes.py
"""

import logging
from datetime import datetime
import json
import ants
import numpy as np
import scipy.ndimage as ni
from aind_exaspim_ccf_reg.configs import VMAX, VMIN
from aind_exaspim_ccf_reg.plots import plot_antsimgs
from skimage.filters import threshold_li
from skimage.measure import label
from typing import List, Optional, Union, Tuple, Dict, Any

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_orientation(
    acquisition_path: str, 
    zarr_image: np.ndarray, 
    logger: logging.Logger
) -> ants.ANTsImage:
    """
    Check and adjust image orientation based on acquisition metadata.
    
    Parameters
    ----------
    acquisition_path : str
        Path to the acquisition metadata JSON file
    zarr_image : np.ndarray
        Input image array
    logger : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    ants.ANTsImage
        Oriented image as ANTs image object
    """
    logger.info("Starting check orientation ......")
    
    with open(acquisition_path, "r") as f:
        metadata = json.load(f)  
        file_name_1st = metadata["tiles"][0]["file_name"]
        logger.info(f"The first tile file name: {file_name_1st}")

        if "tile_000000_ch_" in metadata["tiles"][0]["file_name"]:
            # beta scope
            logger.info("The input is a Beta scope sample!!")
            ccf_directions = {
                0: "Anterior_to_posterior",
                1: "Superior_to_inferior",
                2: "Left_to_right",
            }
        else:
            logger.info("The input is a Alpha scope sample!!")
            ccf_directions = {
                0: "Posterior_to_anterior",
                1: "Inferior_to_superior",
                2: "Left_to_right",
            }

    logger.info(f"CCF_DIRECTIONS {ccf_directions}")
    swaps, flips = get_adjustments(metadata['axes'], ccf_directions)
    logger.info(f"**swaps {swaps}, flips {flips}**")
    zarr_image = adjust_array(zarr_image, swaps, flips)
    ants_img = ants.from_numpy(zarr_image.astype(np.float32))    

    return ants_img


def get_adjustments(
    axes: List[Dict[str, Any]], 
    orientation: Dict[int, str]
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Compute the necessary swaps and flips to adjust the orientation of axes.

    This function compares the given axes with a reference orientation and determines
    the swaps and flips required to match the target orientation.

    Parameters
    ----------
    axes : List[Dict[str, Any]]
        A list of dictionaries containing axis information from the acquisition.json file,
        where each entry includes the 'dimension' and 'direction' of the axis.
        Example format:
        ```
        [
            {
                "name": "X",
                "dimension": 2,
                "direction": "Anterior_to_posterior",
                "unit": "micrometer"
            }
        ]
        ```
    orientation : Dict[int, str]
        A dictionary specifying the reference orientation for each axis dimension.

    Returns
    -------
    Tuple[List[Tuple[int, int]], List[int]]
        A tuple containing:
        - swaps: List of tuples where each tuple contains two dimensions to be swapped
        - flips: List of integers where each integer represents an axis that needs to be flipped
    """
    flips = []
    swaps = []
    
    for i in range(len(axes)):
        ax = axes[i]
        dim = ax["dimension"]
        direction = ax["direction"].lower()

        if orientation[dim].lower() == direction:
            # No change needed
            continue

        for idx, d in orientation.items():
            # Check if only a swap is needed
            if d.lower() == direction:
                swaps.append((dim, idx))
            # Check if flip is needed
            elif d.lower() == "_".join(direction.split("_")[::-1]):
                swaps.append((dim, idx))
                flips.append(idx)

    return swaps, flips


def adjust_array(arr: np.ndarray, swaps: List[Tuple[int, int]], flips: List[int]) -> np.ndarray:
    """
    Adjust a NumPy array by performing axis swaps and flips.

    This function reorders the axes of a NumPy array based on the specified
    swaps, and then flips the array along the specified axes.

    Parameters
    ----------
    arr : np.ndarray
        The input NumPy array to be adjusted.
    swaps : List[Tuple[int, int]]
        A list of tuples representing the axes to be swapped. Each tuple contains
        two dimensions to be swapped.
    flips : List[int]
        A list of integers representing the axes that should be flipped.

    Returns
    -------
    np.ndarray
        The adjusted NumPy array after the swaps and flips are applied.
    """
    if swaps:
        in_axis, out_axis = zip(*swaps)
        arr = np.moveaxis(arr, in_axis, out_axis)
    if flips:
        arr = np.flip(arr, axis=flips)
    return arr


def perc_normalization(
    ants_img: ants.ANTsImage, 
    lower_perc: float = 2, 
    upper_perc: float = 98
) -> ants.ANTsImage:
    """
    Perform percentile normalization on an ANTs image.

    Parameters
    ----------
    ants_img : ants.ANTsImage
        Input ANTs image to normalize
    lower_perc : float
        Lower percentile for normalization (default: 2)
    upper_perc : float
        Upper percentile for normalization (default: 98)

    Returns
    -------
    ants.ANTsImage
        Normalized ANTs image
    """
    percentiles = [lower_perc, upper_perc]
    percentile_values = np.percentile(ants_img.view(), percentiles)
    assert percentile_values[1] > percentile_values[0]
    
    ants_img = (ants_img - percentile_values[0]) / (
        percentile_values[1] - percentile_values[0]
    )

    return ants_img