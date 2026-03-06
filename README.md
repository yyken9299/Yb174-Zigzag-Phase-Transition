# Zigzag Phase Transition in 174Yb+ Ion Chains

## Overview
This repository contains all experimental control and data analysis codes
used in the Master's thesis on the zigzag phase transition of 174Yb+ ion chains.

The repository includes:

1. **Theory codes** (`theory/`) – theoretical calculations and simulations related to the zigzag phase transition and IKZM scaling.
2. **Core algorithms** (`core_algorithms/`) – simplified, clean versions of ion detection and defect analysis algorithms suitable for reproducing thesis results.
3. **Experimental scripts** (`experimental_scripts/`) – full scripts including GUI, batch processing, and file I/O used in the laboratory.
4. **Example data** (`data_examples/`) – optional, small sample datasets for testing purposes.

## Repository Structure

- `theory/` : Theoretical calculations and simulations of the zigzag phase transition.
- `core_algorithms/` : Minimal dependency versions for academic presentation.
- `experimental_scripts/` : Full scripts for experimental use.
- `data_examples/` : Small sample datasets for testing.

## Theory Codes

| Script | Purpose |
|------|------|
| ikzm_scaling_curve.py | Calculate theoretical IKZM scaling curves for defect density versus quench rate |
| phase_transition_simulation.py | Simulate the structural phase transition of trapped ion chains during radial confinement ramp |

## Experimental Scripts Overview

| Script | Purpose |
|--------|---------|
| real_ion_detect.py | Real-time ion monitoring using screen region selection |
| col_tif_all_2guassian.py | Process all TIF images in a single folder and output ion coordinates |
| col_tif_2guassian_forall.py | Batch process entire folder including subfolders |
| col_tif_diff_2guassian.py | Single-image testing of parameter effects without saving files |
| col_tif_check.py | Ion detection check with visualization for different TIF images |
| defect_col_dark_max3.py | Defect analysis for a single folder |
| defect_col_dark_max3_forall.py | Batch defect analysis including subfolders |
| defect_check_dark.py | Defect checking with visualization and marking |

## Usage
- **Core algorithms**: Can be imported and used directly in Python for reproducing thesis results.
- **Experimental scripts**: Designed for batch processing of experimental data in the laboratory.

## Environment
- Python 3.10+
- See `core_algorithms/requirements.txt` for core algorithm dependencies.
- See `experimental_scripts/requirements.txt` for full experimental scripts dependencies.

## License
[MIT License](LICENSE)

## Thesis Version

Release v1.0 corresponds to the version used in the Master's thesis.
