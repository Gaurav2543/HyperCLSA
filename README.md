# Multi-Omics Hypergraph Neural Network Framework

This repository provides two frameworks for multi-omics data analysis:

1. **MORE (Single Train/Test Split):** Uses a predefined training/testing split (via the `train_test` function). This strategy is ideal when you already have a fixed split (e.g., the BRCA dataset).
2. **HyperTMO (Uses 5-Fold Cross-Validation):**
   Uses 5-fold cross-validation to provide a robust evaluation of model performance. This approach is recommended when datasets are limited or heterogeneous (e.g., the ROSMAP dataset).

Both frameworks generate plots (training/testing losses, accuracies, F1 scores, and confusion matrices) and save model weights in designated folders.

---

## Repository Structure

```
.
├── main.py            		# Entry point for single train/test split (e.g., BRCA)
├── train.py           		# Contains training and evaluation functions for cross-validation (e.g., ROSMAP)
├── models.py          		# Model definitions (HGCN, TMO, etc.)
├── utils.py           		# Utility functions (data loading, hypergraph construction, etc.)
├── method_dataset_plots/       # Directory where generated plots are saved
├── data/models/            	# Directory for saving model weights
└── README.md  
```

---

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Install the required packages using:

```bash
pip install torch==1.13.0 numpy==1.24.4 pandas scikit-learn matplotlib
```

## Data Preparation

### MORE (Single Train/Test Split)

For running MORE, create the following directory structure for the dataset:

```
data/
└── dataset/
    ├── labels_tr.csv       # Training labels
    ├── labels_te.csv       # Testing labels
    ├── 1_tr.csv            # Feature view 1 for training
    ├── 1_te.csv            # Feature view 1 for testing
    ├── 2_tr.csv            # Feature view 2 for training
    ├── 2_te.csv            # Feature view 2 for testing
    └── 3_tr.csv            # Feature view 3 for training (if available)
         3_te.csv            # Feature view 3 for testing (if available)
```

### HyperTMO (5-Fold Cross-Validation)

For running HyperTMO, create the following directory structure for the dataset:

```
data/
└── dataset/
    ├── labels.csv          # All sample labels
    ├── miRNA.csv           # miRNA features
    ├── meth.csv            # Methylation features
    └── mRNA.csv            # mRNA expression features
```

Ensure that all CSV files are formatted correctly with comma delimiters.

---

## Running the Code

Run the framework with:

```bash
python main.py --dataset brca --method more --file_dir BRCA --num_class <number_of_classes> --dim_he_list 400 200 200 --view_list 1 2 3 --num_epoch 20000 --lr_e 0.0005 --lr_c 0.001
```

**Notes:**

- `--file_dir`: Path to the dataset folder (e.g., `data/BRCA`).
- `--num_class`: Number of classes in your dataset.
- `--dim_he_list`: Hidden dimensions for the HGCN modules.
- `--view_list`: List of view identifiers (matching your CSV filenames).
- Adjust epochs, learning rates, and other parameters as needed.

**Output:**

- Model weights are saved under `dataset/models/`.
- Plots (including combined training/testing metrics and the confusion matrix) are saved in the `method_dataset_plots/` directory.
