# MFTabPFN
Multi-fidelity tabular prior-data fitted network model

This repository contains the code for MFTabPFN, designed to address accurate prediction and uncertainty quantification for applications using single- or multi-fidelity data. The corresponding datasets for this repository are available at https://doi.org/10.5281/zenodo.16777637.

# Installation
Due to parameter fine-tuning requirements of MFTabPFN, one or more GPUs are required to achieve efficient code execution.

MFTabPFN is developed based on TabPFN 2.2.1. To install MFTabPFN, clone the repository along with its associated datasets. Python 3.11 is required to install the repository. Suppose the conda is used to install, To just set up the MFTabPFN, run the following:
git clone https://github.com/Rockou/MFtabPFN.git
conda create -n MFTabPFN1 python=3.11
cd MFTabPFN
pip install -r requirements1.txt

To setup the MFTabPFN and the compared single-fidelity baselines including RealMLP, TabM, LightGBM, CatBoost, XGBoost, ModernNCA, TorchMLP, TabDPT, EBM, FastaiMLP, ExtraTrees, Autogluon, etc., based on TabArena 0.0.1, run the following:
conda create -n MFTabPFN2 python=3.11
cd MFTabPFN
pip install -r requirements2.txt
pip install -e "TabPFN[dev]"
pip install -e tabpfn-extensions

To set up the linear and nonlinear multi-fidelity Gaussian process regression models, run the following:
conda create -n MFTabPFN3 python=3.11
cd MFTabPFN
pip install -r requirements3.txt

To perform computational fluid dynamics analysis of the DLR-F4 wing-body configuration (turb_DLR_F4.cfg, DLR_F4_mesh1.su2) and the ONERA M6 wing (low-fidelity: inv_ONERAM6_adv_low.cfg; high-fidelity: inv_ONERAM6_adv_high.cfg; mesh file: mesh_ONERAM6_inv_FFD.su2), the SU2 software should be installed. The installation file and installation instructions can be found at the SU2 official website at https://su2code.github.io and github at https://github.com/su2code/SU2. In the article, SU2 v8.2.0 Harrier was used. To facilitate the verification of ML models in computational fluid dynamics application, we have saved the corresponding training and testing datasets computed based on SU2 for the two applications, which can be directly used for various ML models without calling the SU2 software.
# Usage
The repository provides implementations for high-dimensional representation analysis on single-fidelity data (HDR_OpenML_S.py, HDR_Performance_S.py, HDR_Synthetic_S.py, HDR_VaryDataSize_S.py), uncertainty quantification on single-fidelity data (UQ_Prediction_S.py), pressure coefficient distribution prediction for the DLR-F4 wing-body configuration using multi-fidelity data (CP_DLRF4_Position_M.py, CP_DLRF4_Ratio_M.py), and lift and drag coefficient prediction and uncertainty quantification for the ONERA M6 wing based on active learning using multi-fidelity data (CL_ONERAM6_M.py, CD_ONERAM6_M.py). All files include implementations of MFTabPFN and reference machine learning methods. To perform the corresponding analyses, directly execute the respective files. To visualize the results, run the associated plotting files.

Additionally, several demonstration files with toy examples, including Toy_S.py, Toy_M.py, and Toy_Active.py, are provided to illustrate the application of MFTabPFN to single-fidelity data, multi-fidelity data, and active learning, respectively.
