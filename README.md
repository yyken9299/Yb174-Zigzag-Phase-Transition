# Zigzag Phase Transition in 174Yb+ Ion Chains

## Overview
This repository contains all experimental control and data analysis codes
used in the Master's thesis on the zigzag phase transition of 174Yb+ ion chains.

The repository includes:

1. **Core algorithms** (`core_algorithms/`) – simplified, clean versions of ion detection and defect analysis algorithms suitable for reproducing thesis results.
2. **Experimental scripts** (`experimental_scripts/`) – full scripts including GUI, batch processing, and file I/O used in the laboratory.
3. **Theory Simulation** (`theory/`) – Python implementation of the Inhomogeneous Kibble-Zurek Mechanism (IKZM) scaling law.
4. **Example data** (`data_examples/`) – optional, small sample datasets for testing purposes.

## Repository Structure
- `core_algorithms/` : Minimal dependency versions for academic presentation.
- `experimental_scripts/` : Full scripts for experimental use.
- `theory/` : Physical engine for zigzag transition and KZM scaling law scanning.
- `data_examples/` : Small sample datasets for testing.

## Module Details

### Experimental Scripts
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

### Theory Simulation
| Script | Purpose |
|--------|---------|
| zigzag_sim.py | Multiprocessing MD simulation using BAOAB integrator to verify IKZM scaling laws ($d \propto \tau_Q^{-b}$) |

## Usage
- **Theory**: Navigate to `theory/` and run `python zigzag_sim.py`. The script uses a BAOAB integrator to simulate ion cooling and counts structural defects (kinks) across different quench scales ($\tau_Q$).
- **Core algorithms**: Can be imported directly for reproducing thesis results.
- **Core algorithms**: Can be imported and used directly in Python for reproducing thesis results.
- **Experimental scripts**: Designed for batch processing of experimental data in the laboratory.

## Environment
- Python 3.10+
- See `core_algorithms/requirements.txt` for core algorithm dependencies.
- See `theory/requirements.txt` for simulation dependencies.
- See `experimental_scripts/requirements.txt` for full experimental scripts dependencies.

## License
[MIT License](LICENSE)

## Thesis Version
Release v1.0 corresponds to the version used in the Master's thesis.