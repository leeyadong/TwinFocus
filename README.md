# TwinFocus

This code requires a GPU with performance at least comparable to an RTX 4090 to achieve a reasonable
runtime.

## 1. Environment Setup

Create a conda environment and install the required dependencies:

```bash
conda create -n twinfocus python=3.11
conda activate twinfocus
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
pip install kornia matplotlib
```

## 2. Data Download 
Download the dataset from the provided link and extract it into the following directory:
```bash
/TwinFocus/data
```

## 3. Run
```bash
cd code
python autofocus.py
```

The results will be saved in
```bash
/TwinFocus/results
```
