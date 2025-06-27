"""
Main script for exaspim-to-template-to-CCF registration pipeline.

This module contains the main functions for performing CCF (Common Coordinate Framework)
registration of exaSPIM data to the Allen Mouse Brain Atlas.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Optional

import ants
import numpy as np
import zarr
import dask.array as da
from numcodecs import blosc

from pathlib import Path
from aind_exaspim_ccf_reg.utils import (
    create_logger, 
    read_json_as_dict, 
    prepare_config_sample, 
    create_folder, 
    generate_processing,
    extract_dataset_id
)
from aind_exaspim_ccf_reg.configs import PathLike, RegSchema
from aind_exaspim_ccf_reg.preprocess import perc_normalization, check_orientation
from aind_exaspim_ccf_reg.plots import plot_reg, plot_antsimgs
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_exaspim_ccf_reg.register import RegistrationPipeline
from argschema import ArgSchemaParser

__version__ = "0.0.1"
code_url = "https://github.com/AllenNeuralDynamics/aind-exaspim-ccf-registration.git"

def load_zarr(
    image_path: PathLike, 
    logger: logging.Logger
) -> np.ndarray:
    """
    Load Zarr image.
    """
    image = zarr.open(image_path, mode="r")
    image = np.squeeze(np.squeeze(np.array(image), axis=0), axis=0)
    logger.info("----"*10)
    logger.info(f"Loading OMEZarr image from path: {image_path}")
    logger.info(f"image shape: {image.shape}")
    logger.info("----"*10)
        
    return image


def main() -> None:
    """
    Main function to run the CCF registration pipeline.
    
    This function orchestrates the entire registration process:
    1. Loads configuration from processing manifest
    2. Sets up output directories and logging
    3. Performs registration at the specified resolution level
    4. Optionally applies transforms to 10um resolution
    5. Generates processing metadata
    """
    DATA_FOLDER = os.path.abspath("../data")
    RESULTS_FOLDER = os.path.abspath("../results")
    CCF_FOLDER = os.path.abspath(f"{DATA_FOLDER}/allen_mouse_ccf")
    start_time = datetime.now()

    processing_manifest_file = os.path.abspath(f"{DATA_FOLDER}/processing_manifest.json")
    try:
        with open(processing_manifest_file, 'r') as f:
            dataset_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: {processing_manifest_file} not found.")
        return

    print(f"processing_manifest_file: {processing_manifest_file}")

    dataset_path = str(dataset_config["pipeline_processing"]["registration"]["alignment_channel_path"])
    level = int(str(dataset_config["pipeline_processing"]["registration"]["level"]))
    resolution = int(str(dataset_config["pipeline_processing"]["registration"]["resolution"]))
    
    dataset_id = extract_dataset_id(dataset_path)
    print("Dataset ID:", dataset_id)
    
    outprefix_reg = f"{RESULTS_FOLDER}/ccf_alignment/"
    create_folder(dest_dir=outprefix_reg, verbose=True)

    outprefix = f"{RESULTS_FOLDER}/ccf_alignment/registration_metadata/"
    create_folder(dest_dir=outprefix, verbose=True)
    logger = create_logger(output_log_path=outprefix)
    logger.info(f"Processing dataset: {dataset_id}, level={level}, resolution={resolution}um - Output dir: {outprefix} - Config: {dataset_config}")
    logger.info(dataset_path)

    exaspim_to_ccf_transform_path = [
        os.path.abspath("/data/reg_exaspim_template_to_ccf_25um_v1.3/1Warp.nii.gz"),
        os.path.abspath("/data/reg_exaspim_template_to_ccf_25um_v1.3/0GenericAffine.mat")
    ]
    
    logger.info(f"Starting processing for sample {dataset_id}")
    acquisition_output = prepare_config_sample(
        dataset_path=dataset_path,
        logger=logger,
        dataset_id=dataset_id,
        acquisition_output=outprefix,
    )
    logger.info(f"acquisition_output: {acquisition_output}")
    
    example_input = {
        "dataset_path": dataset_path,
        "level": level,
        "resolution": resolution,
        "dataset_id": dataset_id,
        "acquisition_output": acquisition_output, 
        "bucket_path": "aind-scratch-data",
        "outprefix_reg": outprefix_reg,
        "outprefix": outprefix,
        "exaspim_to_ccf_transform_path": exaspim_to_ccf_transform_path,
        "reg_param_25um": {
            "sample_scale": [i/1000.0 for i in [20.25, 20.25, 27]],
            "exaspim_template_path": "../data/exaSPIM_template_25um/exaspim_template_7sujects_nomask_25um_round6.nii.gz",
            "ccf_path": '../data/allen_mouse_ccf/average_template/average_template_25.nii.gz',
            "affine_reg_iterations": [200, 100, 25, 3],
            "syn_reg_iterations": [200, 100, 25, 3],
        },
        "reg_param_10um": {
            "sample_scale": [i/1000.0 for i in [10.125, 10.125, 13.5]],
            "exaspim_template_path": "../data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz",
            "ccf_path": '../data/allen_mouse_ccf/average_template/average_template_10.nii.gz',
            "affine_reg_iterations": [300, 200, 50, 0],
            "syn_reg_iterations": [400, 200, 40, 0],
        }
    }
    data_processes = []
    
    # Create ArgSchemaParser from example_input
    parser = ArgSchemaParser(schema_type=RegSchema, input_data=example_input)
    pipeline = RegistrationPipeline(logger)

    #-----------------------------------------------#
    # Load OMEZarr image 
    #-----------------------------------------------#

    start_date_time = datetime.now()
    image_path = f"{dataset_path}/{level}"
    image = load_zarr(image_path, logger)
    end_date_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_IMPORTING,
            software_version=__version__,
            start_date_time=start_date_time,
            end_date_time=end_date_time,
            input_location=str(image_path),
            output_location=str(image_path),
            outputs={},
            code_url=code_url,
            code_version=__version__,
            parameters={},
            notes="Importing fused data for alignment",
        )
    )

    #-----------------------------------------------#
    # registration
    #-----------------------------------------------#
    start_date_time = datetime.now()

    brain_to_exaspim_transform_path = pipeline.register(
        parser=parser,
        zarr_image=image,
    )
    end_date_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_ATLAS_ALIGNMENT,
            software_version=__version__,
            start_date_time=start_date_time,
            end_date_time=end_date_time,
            input_location=str(image_path),
            output_location=outprefix_reg,
            outputs={},
            code_url=code_url,
            code_version=__version__,
            parameters={},
            notes="Template based registration: sample -> template -> CCF",
        )
    )

    if level in [3, 6]:
        level = {3: 2, 6: 5}.get(level, level)    
        resolution = 10

        #-----------------------------------------------#
        # load zarr
        #-----------------------------------------------#
        start_date_time = datetime.now()
        image_path = f"{dataset_path}/{level}"
        image = load_zarr(image_path, logger)    
        end_date_time = datetime.now()
        data_processes.append(
            DataProcess(
                name=ProcessName.IMAGE_IMPORTING,
                software_version=__version__,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                input_location=str(image_path),
                output_location=str(image_path),
                outputs={},
                code_url=code_url,
                code_version=__version__,
                parameters={},
                notes="Importing fused data for alignment",
            )
        )

        #-----------------------------------------------#
        # apply transforms to 10um
        #-----------------------------------------------#    
        start_date_time = datetime.now()
        output_path = pipeline.apply_transforms_to_10um(
            parser=parser,
            zarr_image=image,
            brain_to_exaspim_transform_path=brain_to_exaspim_transform_path,
            dataset_id=f"{dataset_id}_10um",
        )

        end_date_time = datetime.now()
        data_processes.append(
            DataProcess(
                name=ProcessName.IMAGE_ATLAS_ALIGNMENT,
                software_version=__version__,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                input_location=str(image_path),
                output_location=str(output_path),
                outputs={},
                code_url=code_url,
                code_version=__version__,
                parameters={},
                notes=f"Registering 10um sample to CCF using transforms: {brain_to_exaspim_transform_path} and {exaspim_to_ccf_transform_path}",
            )
        )  

    processing_path = Path(outprefix_reg).joinpath("processing.json")

    logger.info(f"Writing processing: {processing_path}")

    generate_processing(
        data_processes=data_processes,
        dest_processing=outprefix_reg,
        processor_full_name="Di Wang",
        pipeline_version="0.0.1",
    )

    end_time = datetime.now()    
    logger.info(f"Finish all steps, execution time: {end_time - start_time} s")


if __name__ == "__main__":
    main()
