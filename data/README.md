# Dataset

This project is based on the processed Z24 Bridge Benchmark dataset.

The original dataset can be downloaded from:

https://huggingface.co/datasets/thanglexuan/Z24-dataset-processed

After downloading, place the following files into this directory:

```text
data/
├── inputs.npy
└── labels.npy
```

These files are not included in the repository because they exceed GitHub's file size limit.

## Precomputed Features

This repository includes the file:

```text
X_features_frequency_curvature.npy
```

This file contains the precomputed feature vectors extracted from the raw vibration signals using Frequency Domain Decomposition (FDD). Each feature vector consists of the modal frequencies and modal curvature features for one sample and serves as the input to the machine learning models.

Users can either:

- use the provided `X_features_frequency_curvature.npy` directly to reproduce the machine learning experiments, or
- regenerate the feature file by running `src/FDD.py` after downloading the original dataset.