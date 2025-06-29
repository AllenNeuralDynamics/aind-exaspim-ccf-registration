# exaSPIM-to-CCF Registration Pipeline

A pipeline for registering exaSPIM data to the Allen Mouse Common Coordinate Framework (CCF).

## Overview

This pipeline performs automated registration of exaSPIM images to CCF via a exaSPIM template. The registration process involves:
1. **Image Loading**: Loading OME-Zarr exaSPIM data.
2. **Preprocessing**: Image normalization and orientation checking.
3. **Registration**: Multi-resolution registration using ANTs (Advanced Normalization Tools).
4. **Transform Application**: Applying registration transforms to high-resolution data (10um).
5. **Output Generation**: Saving registered images and metadata.



## Usage

### Basic Usage

1. **Prepare your data structure**:
```
data/
├── processing_manifest.json
├── allen_mouse_ccf/
│   └── average_template/
│       └── average_template_25.nii.gz
│── exaSPIM_template_25um/
│   └── exaspim_template_7sujects_nomask_25um_round6.nii.gz
└── exaSPIM_template_to_ccf_transforms/
```

2. **Run the pipeline**:
```bash
python main.py
```

### Configuration

The pipeline uses a `processing_manifest.json` file to configure the registration process:

```json
{
  "pipeline_processing": {
    "registration": {
      "alignment_channel_path": "/path/to/your/ome-zarr/dataset",
      "level": 3,
      "resolution": 25
    }
  }
}
```

#### Configuration Parameters

- `alignment_channel_path`: S3 path to the OME-Zarr dataset
- `level`: Processing level (typically 3 for 25μm, 2 for 10μm)
- `resolution`: Resolution in microns (25 or 10)

### Output Structure

The pipeline generates the following output structure:

```
results/
└── ccf_alignment/
    ├── registration_metadata/
    │   ├── logs/
    │   ├── intermediate images
    │   └── registration visualization
    ├── ccf_aligned.zarr
    ├── ccf transforms
    └── processing.json
```

## Pipeline Components

### Core Modules

- **`main.py`**: Main pipeline orchestration
- **`aind_exaspim_ccf_reg/`**: Core registration modules
  - **`register.py`**: Registration pipeline implementation
  - **`preprocess.py`**: Image preprocessing functions
  - **`configs.py`**: Configuration schemas and types
  - **`utils.py`**: Utility functions

### Registration Process

1. **Image Loading**: Loads OME-Zarr data at specified resolution
2. **Preprocessing**: 
   - Percentile normalization
   - Orientation checking and correction
3. **Sample-to-Template Registration**:
   - Affine registration
   - SyN registration
4. **High-resolution Processing** (10μm if applicable):
   - Load 10μm data
   - Apply Sample-to-Template transforms from 25μm registration
   - Apply Template-to-CCF transforms
5. **Output Generation**:
   - Save registered images
   - Generate processing metadata
   
   
   
## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```
black .
```

- Use **isort** to automatically sort import statements:
```
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect the build system or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests
