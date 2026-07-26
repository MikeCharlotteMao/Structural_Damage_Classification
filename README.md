# A Machine-Learning-Based Framework for Multi-Class Structural Damage Classification Using Frequency–Curvature Modal Features

## Overview

This repository contains the implementation of a machine-learning-based framework for multi-class structural damage classification using modal frequencies and modal curvatures extracted from vibration responses.

Frequency Domain Decomposition (FDD) is first applied to identify modal frequencies and mode shapes from structural vibration data. Modal curvatures are subsequently computed and combined with modal frequencies to construct feature vectors. Three tree-based machine learning algorithms are then optimized and evaluated for structural damage classification.

The implemented models include:

- Random Forest (RF)
- Extreme Gradient Boosting (XGBoost)
- Extremely Randomized Trees (Extra Trees)

The framework is evaluated on the Z24 Bridge Benchmark dataset containing 17 structural conditions.

---

## Repository Structure

```
STRUCTURAL_DAMAGE_CLASSIFICATION
│
├── data/
│   ├── inputs.npy
│   └── labels.npy
│
├── results/
│   ├── figures
│   ├── confusion matrices
│   ├── feature-selection results
│   └── hyperparameter search results
│
├── src/
│   ├── __init__.py
│   ├── FDD.py
│   ├── FDD_explore.py
│   ├── RF_FeatureImportance.py
│   ├── ExtraTrees_FeatureImportance.py
│   └── XGBoost_FeatureImportance.py
│
├── README.md
└── requirements.txt
```

---

## Dataset

The project uses the processed Z24 Bridge Benchmark dataset.

```
inputs.npy
```

- Structural acceleration responses
- Shape: `(1530, 27, 6000)`

```
labels.npy
```

- Damage labels
- 17 structural conditions
- Shape: `(1530,)`

The processed dataset is publicly available at:

https://huggingface.co/datasets/thanglexuan/Z24-dataset-processed

The dataset is not included in this repository because the `.npy` files exceed GitHub's file size limit.
---

## Source Code

### FDD.py

Performs Frequency Domain Decomposition (FDD) on a single vibration sample.

The script

- computes the cross-power spectral density matrix,
- performs Singular Value Decomposition,
- extracts the First Singular Value spectrum,
- identifies modal frequencies,
- estimates corresponding mode shapes.

---

### FDD_explore.py

Explores modal frequency distributions across representative samples.

This script

- performs FDD on multiple samples,
- detects frequency peaks,
- identifies the most populated modal frequency regions,
- determines the frequency bands used for feature extraction.

---

### RF_FeatureImportance.py

Optimizes the Random Forest classifier.

The script evaluates

- Top 20 features
- Top 30 features
- Top 40 features
- Top 50 features
- Top 60 features
- Full 78 features

For each feature subset, multiple hyperparameter combinations are tested and evaluated using

- Accuracy
- Precision
- Recall
- F1-score

The optimal feature subset and hyperparameter combination are selected according to the evaluation metrics.

---

### XGBoost_FeatureImportance.py

Performs feature selection and hyperparameter optimization for the XGBoost classifier following the same workflow as Random Forest.

---

### ExtraTrees_FeatureImportance.py

Performs feature selection and hyperparameter optimization for the Extra Trees classifier following the same workflow.

---

## Results

The `results/` directory contains

- Feature-selection performance curves
- Hyperparameter search results
- Confusion matrices
- FDD visualization
- Peak-frequency distribution
- Model comparison figures
- Performance tables

---

## Performance

The best-performing models obtained in this study are

| Model         | Best Features   | Accuracy  |
| ------------- | --------------- | --------- |
| Random Forest | Top 20          | 0.846     |
| XGBoost       | Top 40          | 0.853     |
| Extra Trees   | Top 20 / Top 30 | **0.863** |

Extra Trees achieved the highest overall classification performance.

---

## Requirements

Install the required packages using

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this repository, please cite

Lecheng Mao,
*A Machine-Learning-Based Framework for Multi-Class Structural Damage Classification Using Frequency–Curvature Modal Features*.

