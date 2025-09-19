# HyperCLSA: Hypergraph Contrastive Learning with Self-Attention for Multi-Omics Integration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

HyperCLSA is a novel deep learning framework for multi-omics data integration in cancer subtyping. It combines hypergraph-based sample encoding, supervised contrastive learning for latent space alignment, and multi-head self-attention for adaptive fusion of omics modalities. This repository contains the implementation for our accepted paper on breast cancer PAM50 subtype classification using TCGA-BRCA dataset.

### Key Features

* **Hypergraph Neural Networks** : Captures higher-order relationships between samples using HGNN encoders
* **Supervised Contrastive Learning** : Aligns multi-modal representations in a shared latent space
* **Multi-Head Self-Attention** : Adaptively fuses information across different omics modalities
* **Advanced Feature Selection** : Supports Boruta, RFE, and SHAP-based feature selection methods
* **Cross-Validation** : Robust evaluation using stratified k-fold cross-validation
* **Flexible Architecture** : Modular design adaptable to different cancer types and datasets

## Repository Structure

```
HYPERCLSA/
├── src/                          # Source code directory
│   ├── train.py                 # Main training script with cross-validation
│   ├── models.py                # HyperCLSA model architecture
│   ├── losses.py                # Contrastive loss implementation
│   ├── utils.py                 # Utility functions and data loading
│   ├── feature_selection.py     # Feature selection methods (Boruta, RFE, SHAP)
│   ├── graph_utils.py          # Hypergraph construction utilities
│   ├── main.py                 # Main execution script
│   └── tune_optuna.py          # Hyperparameter tuning with Optuna
├── run_outputs/                 # Training output files
│   ├── best_meth_view3.out
│   ├── best_miRNA_view2.out
│   ├── best_mRNA_view1.out
│   ├── best_run.out
│   ├── best_view1_2.out
│   ├── best_view1_3.out
│   └── main_new_best_view2_3.out
├── Other_Tools/                 # Additional tools and baselines
│   ├── HyperTMO/               # HyperTMO baseline implementation
│   └── MORE/                   # MORE baseline implementation
├── oldest_results.md           # Historical results documentation
├── best_params.json           # Optimized hyperparameters
├── plot.ipynb                 # Results visualization notebook
└── README.md                  # This file
```

## Installation

### Requirements

* Python 3.9+
* PyTorch 1.13+
* CUDA-compatible GPU (recommended)

### Dependencies

Install the required packages using pip:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install boruta shap
pip install optuna
pip install matplotlib seaborn
pip install jupyter notebook
```

Or install from requirements file (if available):

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Prepare your multi-omics data in CSV format with the following structure:

* `labels_tr.csv` and** **`labels_te.csv`: Training and testing labels
* `{view}_tr.csv` and** **`{view}_te.csv`: Training and testing data for each omics view
* `{view}_featname.csv`: Feature names for each view (optional, for feature selection)

Example directory structure for TCGA-BRCA dataset:

```
ROSMAP/  # or your dataset directory
├── labels_tr.csv
├── labels_te.csv
├── 1_tr.csv        # mRNA training data
├── 1_te.csv        # mRNA testing data
├── 2_tr.csv        # miRNA training data
├── 2_te.csv        # miRNA testing data
├── 3_tr.csv        # methylation training data
├── 3_te.csv        # methylation testing data
├── 1_featname.csv  # mRNA feature names
├── 2_featname.csv  # miRNA feature names
└── 3_featname.csv  # methylation feature names
```

### 2. Basic Usage

Run HyperCLSA with default parameters:

```bash
cd src
python main.py
```

### 3. Custom Configuration

Modify the parameters in** **`main.py` or create your own configuration:

```python
from train import train_test_CLCLSA
from utils import set_seed

# Set your parameters
params = {
    "graph_method": "radius",
    "k_neigs": 6,
    "radius_eps": 0.696573230951862,
    "latent_dim": 64,
    "attn_heads": 1,
    "hidden1": 400,
    "hidden2": 256,
    "lr": 0.0018717267107673911,
    "lambda_contrast": 0.29547396996824615,
    "fs_method": "boruta",
    "boruta_max_iter": 60
}

# Run training and evaluation
set_seed(42)
results = train_test_CLCLSA(
    data_folder="your_dataset",
    view_list=[1, 2, 3],
    num_class=5,
    **params,
    n_splits_cv=5
)
```

## Configuration Options

### Model Architecture

* `hidden_dims`: Hidden layer dimensions for HGNN encoders (default: [400, 256])
* `latent_dim`: Shared latent space dimension (default: 64)
* `attn_heads`: Number of attention heads (default: 1)

### Training Parameters

* `lr`: Learning rate (default: 0.00187)
* `epochs`: Maximum training epochs (default: 5000)
* `lambda_contrast`: Weight for contrastive loss (default: 0.295)

### Hypergraph Construction

* `graph_method`: Method for hypergraph construction ("knn", "radius", "mutual_knn")
* `k_neigs`: Number of neighbors for k-NN methods (default: 6)
* `radius_eps`: Radius threshold for radius-based method (default: 0.697)

### Feature Selection

* `fs_method`: Feature selection method ("boruta", "rfe", "shap", or None)
* `boruta_max_iter`: Maximum iterations for Boruta (default: 60)

## Advanced Usage

### Hyperparameter Optimization

Use Optuna for automated hyperparameter tuning:

```bash
cd src
python tune_optuna.py
```

The optimization will search for the best combination of:

* Graph construction parameters
* Model architecture parameters
* Training hyperparameters
* Feature selection settings

### Feature Selection Methods

#### Boruta (Recommended)

```python
fs_kwargs = {"max_iter": 60}
results = train_test_CLCLSA(..., fs_method="boruta", fs_kwargs=fs_kwargs)
```

#### Recursive Feature Elimination (RFE)

```python
fs_kwargs = {"k": 500, "step": 0.1}
results = train_test_CLCLSA(..., fs_method="rfe", fs_kwargs=fs_kwargs)
```

#### SHAP-based Selection

```python
fs_kwargs = {"k": 500}
results = train_test_CLCLSA(..., fs_method="shap", fs_kwargs=fs_kwargs)
```

### Cross-Validation

The framework supports stratified k-fold cross-validation:

```python
results = train_test_CLCLSA(
    ...,
    n_splits_cv=5  # 5-fold CV
)

print(f"Mean F1-Macro: {results['mean_f1_macro']:.4f} ± {results['std_f1_macro']:.4f}")
print(f"Mean Accuracy: {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
```

## Results Visualization

Use the provided Jupyter notebook for results analysis:

```bash
jupyter notebook plot.ipynb
```

The notebook includes:

* Performance comparison plots
* Ablation study visualizations
* Feature importance analysis
* Cross-validation results

## Evaluation Metrics

HyperCLSA reports the following metrics:

* **Accuracy** : Overall classification accuracy
* **F1-Macro** : Macro-averaged F1-score (accounts for class imbalance)
* **F1-Weighted** : Weighted-averaged F1-score

Results are reported as mean ± standard deviation across cross-validation folds.

## Performance Results

On TCGA-BRCA dataset (PAM50 subtype classification):

| Method              | Accuracy               | F1-Macro               | F1-Weighted            |
| ------------------- | ---------------------- | ---------------------- | ---------------------- |
| MOGONET             | 0.829±0.018           | 0.774±0.017           | 0.825±0.016           |
| MORE                | 0.835±0.020           | 0.768±0.021           | 0.820±0.023           |
| HyperTMO            | 0.858±0.023           | 0.821±0.019           | 0.863±0.023           |
| **HyperCLSA** | **0.901±0.007** | **0.866±0.019** | **0.901±0.007** |

## Ablation Studies

The framework includes comprehensive ablation studies:

1. **Feature Selection Impact** : Boruta vs RFE vs No FS
2. **Hypergraph Construction** : k-NN vs Radius vs Mutual k-NN
3. **Attention Mechanisms** : Self-attention vs simple aggregation
4. **Multi-Omics Integration** : Individual vs combined modalities

## Troubleshooting

### Common Issues

1. **Feature Selection Takes Too Long**
   * Reduce** **`boruta_max_iter` parameter
   * Use RFE with smaller feature counts
   * Consider pre-filtering features by variance
2. **Poor Performance**
   * Check data quality and preprocessing
   * Verify label distribution and class balance
   * Adjust hyperparameters using Optuna
3. **Import Errors**
   * Ensure all dependencies are installed
   * Check Python and PyTorch versions
   * Verify CUDA compatibility

### Performance Optimization

1. **Memory Optimization**

   * Use mixed precision training
   * Implement gradient accumulation for large batches
   * Clear cache between folds
2. **Speed Optimization**

   * Use DataLoader with multiple workers
   * Enable CUDA optimizations
   * Consider model pruning for deployment

### Citation

If you use HyperCLSA in your research, please cite our paper:

```bibtex
@article{hyperclsa2025,
  title={Breast Cancer Subtyping with HyperCLSA: A Hypergraph Contrastive Learning Pipeline for Multi-Omics Data Integration},
  author={Gaurav Bhole, Poorvi HC, Madhav J, Vinod PK, Prabhakar Bhimalapuram},
  conference={11th International Conference on Pattern Recognition and Machine Intelligence, Delhi},
  year={2025},
}
```

## Acknowledgments

* TCGA Research Network for providing the BRCA dataset
* The open-source community for the foundational libraries
* Reviewers and collaborators for their valuable feedback

---

 **Note** : This implementation is provided for research purposes. For clinical applications, please consult with domain experts and follow appropriate validation procedures.
