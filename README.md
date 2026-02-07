# MFTabPFN

**Multi-Fidelity Tabular Prior-Data Fitted Network**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- If you have an arXiv / paper link, replace the line below -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-preprint-red?logo=arXiv)](https://arxiv.org/abs/...) -->

MFTabPFN is an advanced extension of TabPFN, specifically designed for accurate prediction and uncertainty quantification on both single-fidelity and multi-fidelity tabular data. It is useful in data-scarce regimes, with key applications in computational fluid dynamics (CFD), active learning, and other high-precision domains.

Datasets used in this work are publicly available at:  
🔗 [Zenodo DOI: 10.5281/zenodo.18502924](https://doi.org/10.5281/zenodo.18502924)

## ✨ Highlights

- Built on TabPFN v2.2.1 with multi-fidelity extensions
- GPU-accelerated fine-tuning and inference
- Comprehensive baselines from TabArena v0.0.1 (RealMLP, TabM, LightGBM, XGBoost, CatBoost, AutoGluon, ModernNCA, TabDPT, EBM, FastaiMLP, ExtraTrees, etc.)
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
conda create -n mftabpfn python=3.11
conda activate mftabpfn
cd MFTabPFN
pip install -r requirements1.txt
```
This takes 2-3 minutes to build (excluding the time to clone the repository). After installation, you can verify it works by running one of the toy examples:

```bash
cd Test_example
python Toy_S.py
```
The runtime for this toy example is expected to be 20-30 seconds, with the expected output shown in the Toy_S_plot file. 

### 2. Full Setup with TabArena Baselines

Includes RealMLP, TabM, LightGBM, CatBoost, XGBoost, ModernNCA, TabDPT, EBM, FastaiMLP, ExtraTrees, AutoGluon, etc.

```bash
conda create -n mftabpfn-full python=3.11
conda activate mftabpfn-full
cd MFTabPFN
pip install -r requirements2.txt
pip install -e "TabPFN[dev]"
pip install -e tabpfn-extensions
```
This takes 3-4 minutes to build.

### 3. Multi-Fidelity Gaussian Process Baselines

For linear and nonlinear multi-fidelity GPR models.

```bash
conda create -n mftabpfn-gpr python=3.11
conda activate mftabpfn-gpr
cd MFTabPFN
pip install -r requirements3.txt
```
This takes 0.5-1 minute to build.

### CFD Simulations (Optional)

To reproduce SU2-based CFD datasets (DLR-F4 wing-body and ONERA M6 wing):

- Install SU2 v8.2.0 Harrier → Official Site: https://su2code.github.io   GitHub: https://github.com/su2code/SU2
- Pre-computed training/testing datasets are included in the Zenodo archive — no need to run SU2 yourself.

## Quick Start & Usage

The repository is organized into three main application directories:

- Single_fidelity_application — Prediction & UQ on single-fidelity data
- Multi_fidelity_application — Pressure coefficient prediction for DLR-F4 wing-body (multi-fidelity)
- Active_learning — Drag coefficient prediction & UQ for ONERA M6 wing with active learning (multi-fidelity)

Each directory contains MFTabPFN implementation, baseline methods, and visualization tools.

Run an example (single-fidelity):

```bash
cd Single_fidelity_application
python UQ_Prediction_S.py  # Run the model
python UQ_S_Plot.py        # Visualize results
```

Toy examples (quick demonstrations):

```bash
cd Test_example
python Toy_S.py       # Single-fidelity
python Toy_M.py       # Multi-fidelity
python Toy_Active.py  # Active learning with multi-fidelity
```

Results (metrics, predictions, figures) are automatically saved in the working directory or subfolders.

---
Last updated: February 2026
