"""
This config file points to data directories, defines global variables,
specify schema format for Preprocess and Registration.
"""

from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
from argschema import ArgSchema
from argschema.fields import Dict as sch_dict
from argschema.fields import Int, Float
from argschema.fields import List as sch_list
from argschema.fields import Str
from argschema.fields import Nested

PathLike = Union[str, Path]
ArrayLike = Union[da.core.Array, np.ndarray]

VMIN = 0
VMAX = 1.5


class RegParamSchema(ArgSchema):
    sample_scale = sch_list(
        cls_or_instance=Float,
        metadata={"required": True, "description": "Sample scale factors (list of floats)"}
    )
    exaspim_template_path = Str(
        metadata={"required": True, "description": "Path to exaspim template"}
    )
    ccf_path = Str(
        metadata={"required": True, "description": "Path to CCF template"}
    )
    affine_reg_iterations = sch_list(
        cls_or_instance=Int,
        metadata={"required": True, "description": "Affine registration iterations"}
    )
    syn_reg_iterations = sch_list(
        cls_or_instance=Int,
        metadata={"required": True, "description": "SyN registration iterations"}
    )


class RegSchema(ArgSchema):
    dataset_path = Str(required=True, metadata={"description": "Path to dataset"})
    level = Int(required=True, metadata={"description": "input data level (int)"})
    resolution = Int(required=True, metadata={"description": "Resolution (int)"})
    dataset_id = Str(required=True, metadata={"description": "Dataset ID"})
    bucket_path = Str(required=True, metadata={"description": "Bucket path"})
    outprefix_reg = Str(required=True, metadata={"description": "Output prefix for registration"})
    outprefix = Str(required=True, metadata={"description": "Output prefix for metadata"})
    exaspim_to_ccf_transform_path = sch_list(
        cls_or_instance=Str,
        required=True,
        metadata={"description": "List of exaspim template-to-ccf transform paths"}
    )
    reg_param_25um = Nested(RegParamSchema, required=True)
    reg_param_10um = Nested(RegParamSchema, required=True)
    acquisition_output = Str(required=False, metadata={"description": "Path to acquisition metadata"})