---
language: en
tags:
- tabular
- transformer
- single- and multi-fidelity
- uncertainty quantification
- active learning
- few-shot
license: apache-2.0  # 请根据实际情况修改
---

# MFTabPFN: Multi-Fidelity Tabular Prior-Data Fitted Network

**Model Card** for the paper:  
**"A multi-fidelity tabular prior-data fitted network model for accurate prediction and uncertainty quantification"**  
Nature Portfolio submission (2025)

## Model Details

- **Model name**: MFTabPFN (Multi-Fidelity Tabular Prior-Data Fitted Network)
- **Developers**: Yan Shi¹, Cheng Liu¹ et al. (see full author list in the paper)
- **Version**: v1.0 (March 2025)
- **Base model**: TabPFN v2.2.1 (transformer-based prior-data fitted network)
- **Paper**: [arXiv / journal link will be added after acceptance]
- **Repository**: https://github.com/Rockou/MFTabPFN
- **Pretrained models**: Available in `/checkpoints/` folder

### Architecture
MFTabPFN extends TabPFN with a hierarchical transformer architecture:
- **Low-fidelity model**: Pre-trained TabPFN (frozen parameters) processing low-fidelity data.
- **High-fidelity model**: Lightweight encoder (RCNN for high-dimensional data; MLP for low-dimensional) + trainable scaling factor α appended to TabPFN.
- **Final prediction**: Ŷ = α·Z_low(X) + Z_high(X)
- **Output**: Full predictive distribution (PDF) enabling uncertainty quantification (epistemic UQ via prediction variance).

Supports both **single-fidelity** and **multi-fidelity** tabular data seamlessly.

## Intended Uses

- Accurate regression prediction and uncertainty quantification in **data-scarce** regimes (few-shot / low-data settings).
- Multi-fidelity scenarios where high-fidelity data are expensive (e.g., CFD simulations, experiments).
- Scientific computing applications: Computational Fluid Dynamics (DLR-F4 wing-body, ONERA M6 wing), engineering performance functions, benchmark tabular datasets (forest fire, wine quality, etc.).
- Active learning framework (AMFTabPFN) for resource-constrained experimental/simulation design.

**Not intended for**: 
- High-stakes medical or legal decisions without human oversight.
- Classification tasks (regression only in current version).

## Training Procedure

- Low-fidelity component: Uses official pre-trained TabPFN v2.2.1 weights (no further pre-training).
- High-fidelity encoder: Fine-tuned with frozen TabPFN backbone.
- Optimizer: AdamW (default hyperparameters in code).
- Loss: Mean squared error on scaled discrepancy.
- Hardware: NVIDIA RTX A6000 / similar GPU.

## Evaluation

MFTabPFN consistently outperforms state-of-the-art baselines (TabPFN, AutoGluon, RealMLP, TabM, MFGPR, etc.) on:
- 23 single-fidelity OpenML-CTR23 + synthetic high-dimensional datasets.
- Real multi-fidelity CFD cases (DLR-F4 pressure coefficient, ONERA M6 drag coefficient PDF).
- Metrics: NNRMSE, NNMAE, R², RMSE, MAE, JS divergence & Wasserstein-1 (for UQ).

Full results and ablation studies are in the paper and Supplementary Information.

## Limitations

- Requires GPU for best performance (TabPFN backbone).
- Currently focused on regression tasks.
- Interpretability not explicitly studied (mechanism validation of cross-fidelity correction is provided via Pearson correlation and cosine similarity in DLR-F4 application).
- Performance on extremely high-dimensional data (>500 features) not yet benchmarked.

## Ethical Considerations

This model is intended for scientific and engineering research. Users should verify predictions with domain experts when used in safety-critical applications.

## Citation

```bibtex
@article{shi2025mftabpfn,
  title={A multi-fidelity tabular prior-data fitted network model for accurate prediction and uncertainty quantification},
  author={Shi, Yan and Liu, Cheng and et al.},
  journal={Nature [Portfolio Journal]},
  year={2025}
}
