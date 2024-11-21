# SurfPatch: Enabling Patch Matching for Exploratory Stream Surface Visualization

[Delin An](https://github.com/adlsn) and [Chaoli Wang](https://sites.nd.edu/chaoli-wang/)

University of Notre Dame<sup>1</sup>

<div align='center'>
<img src='video.gif'>
</div>

## Introduction
> SurfPatch is a framework for exploratory stream surface visualization. Our method leverages a hierarchical, bottom-up approach with vertex-level classification, patch-level matching, and surface-level clustering for fine-grained multiscale analysis. SurfPatch supports stream surfaces from steady and unsteady flows and isosurfaces from scalar fields, providing an intuitive interface for users to explore and analyze various surfaces.
---
## Architecture overview of SurfPatch
> SurfPatch is for analyzing surfaces through a three-stage process: (1) classifying vertices based on heat kernel signature (HKS) features and partitioning the surface into fine-grained patches using agglomerative hierarchical clustering (AHC) with connectivity constraints, (2) matching similar patches within or across surfaces by aggregating vertex-level features into patch-level features, and (3) clustering surfaces by further aggregating patch-level features into surface-level features, enabling efficient querying and exploration of patches and surfaces.
<div align='center'>
<img src='framework.png'>
</div>

## Installation
The code is developed by Python. After cloning the repository, follow the steps below for installation:
1. Create and activate the conda environment
```python
conda create --name surfpatch python=3.10
conda activate surfpatch
```
2. Install dependencies
```python
pip install -r requirements.txt
```
3. Run the interface
```python
python main.py
```

## Dataset
SurfPatch could handle stream surfaces generated from steady and unsteady flows and isosurfaces extracted from scalar fields. The dataset folder for surfaces should be organized as follows:
```
./dataset/
|
├── data/
├── gallery/
├── hks_feature/
├── patch/
│   └── level1/
└── project2d/
    └── level1/
```
We provide a subset of the [Two Swirls dataset](https://drive.google.com/drive/folders/1-PPu12Ls-kHmZtfqaNW3rfJ199-R4bpg?usp=drive_link) as an example for testing our method.

## Contact
Should you have any questions, please send emails to dan3@nd.edu.











