# MFTabPFN

**Multi-Fidelity Tabular Prior-Data Fitted Network**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- If you have an arXiv / paper link, replace the line below -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-preprint-red?logo=arXiv)](https://arxiv.org/abs/...) -->

MFTabPFN is an advanced extension of TabPFN, specifically designed for accurate prediction and uncertainty quantification on both single-fidelity and multi-fidelity tabular data. It delivers strong performance in data-scarce regimes, with key applications in computational fluid dynamics (CFD), active learning, and other high-precision domains.

Datasets used in this work are publicly available at:  
🔗 [Zenodo DOI: 10.5281/zenodo.16777637](https://doi.org/10.5281/zenodo.16777637)

## ✨ Highlights

- Superior performance on single- and multi-fidelity tabular tasks
- Built on TabPFN v2.2.1 with multi-fidelity extensions
- GPU-accelerated fine-tuning and inference
- Comprehensive baselines from TabArena (RealMLP, TabM, LightGBM, XGBoost, CatBoost, AutoGluon, ModernNCA, TabDPT, EBM, FastaiMLP, ExtraTrees, etc.)
- Ready-to-run examples for CFD (DLR-F4 wing-body, ONERA M6 wing) and toy demonstrations

## Requirements

- Python = 3.11
- One or more GPUs (strongly recommended for efficient fine-tuning)
- Git + Conda (or equivalent environment manager)

## Installation

Three separate environments are provided depending on your use case.

### 1. Core MFTabPFN (minimal setup for the main model)

```bash
git clone https://github.com/Rockou/MFTabPFN.git
conda create -n mftabpfn python=3.11 -y
conda activate mftabpfn
cd MFTabPFN
pip install -r requirements1.txt
```
After installation, you can verify it works by running one of the toy examples:

```bash
cd Test_example
python Toy_S.py
```

### 2. Full Setup with TabArena Baselines

Includes RealMLP, TabM, LightGBM, CatBoost, XGBoost, ModernNCA, TabDPT, EBM, FastaiMLP, ExtraTrees, AutoGluon, etc.

```bash
conda create -n mftabpfn-full python=3.11 -y
conda activate mftabpfn-full

cd MFTabPFN

pip install -r requirements2.txt
pip install -e "TabPFN[dev]"
pip install -e tabpfn-extensions
```

### 3. Multi-Fidelity Gaussian Process Baselines

For linear and nonlinear multi-fidelity GPR models.

conda create -n mftabpfn-gpr python=3.11 -y
conda activate mftabpfn-gpr

cd MFTabPFN

pip install -r requirements3.txt

### CFD Simulations (Optional)

To reproduce SU2-based CFD datasets (DLR-F4 wing-body and ONERA M6 wing):

- Install SU2 v8.2.0 Harrier → Official Site: https://su2code.github.io   GitHub: https://github.com/su2code/SU2
- Pre-computed training/testing datasets are included in the Zenodo archive — no need to run SU2 yourself.

## Quick Start & Usage

The repository is organized into three main application directories:

- Single_fidelity_application/ — Prediction & UQ on single-fidelity data
- Multi_fidelity_application/ — Pressure coefficient prediction for DLR-F4 wing-body (multi-fidelity)
- Active_learning/ — Drag coefficient prediction & UQ for ONERA M6 wing with active learning (multi-fidelity)

Each directory contains MFTabPFN implementation, baseline methods, main scripts, and visualization tools.

Run an example (single-fidelity):

cd Single_fidelity_application
python TabPFN_model.py           # Run the model
python HDR_Performance_S_Plot.py # Visualize results

Toy examples (quick demonstrations):

cd Test_example
python Toy_S.py       # Single-fidelity
python Toy_M.py       # Multi-fidelity
python Toy_Active.py  # Active learning with multi-fidelity

Results (metrics, predictions, figures) are automatically saved in the working directory or subfolders.

## Citation

If you find MFTabPFN useful in your research, please cite:

@article{YAN202X_MFTabPFN,
  title={Multi-Fidelity Tabular Prior-Data Fitted Network for Accurate Prediction and Uncertainty Quantification},
  author={YAN et al.},
  journal={...},
  year={202X},
  doi={...}
}

Also consider citing the foundational work:

@article{hollmann2023tabpfn,
  title={TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Hollmann, Noah and M{\"u}ller, Samuel and H{\"o}ning, Robin and Kerschbaum, Frank and Hutter, Frank},
  journal={arXiv preprint arXiv:2207.01848},
  year={2023}
}

## License

MIT License — see the LICENSE file for details.  
Model weights and datasets may be subject to separate terms (refer to Zenodo).

---
Last updated: February 2026
