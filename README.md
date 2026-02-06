# MFTabPFN

**Multi-Fidelity Tabular Prior-Data Fitted Network**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

<!-- 如果有论文链接，请替换下面的 arXiv 徽章 -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-preprint-red?logo=arXiv)](https://arxiv.org/abs/...) -->

MFTabPFN is an advanced extension of TabPFN, specifically designed for **accurate prediction** and **uncertainty quantification** on both single-fidelity and multi-fidelity tabular data. It delivers great performance in data-scarce regimes, with strong applications in computational fluid dynamics (CFD), active learning, and other high-precision domains.

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
cd MFTabPFN

conda create -n mftabpfn python=3.11 -y
conda activate mftabpfn

pip install -r requirements1.txt
