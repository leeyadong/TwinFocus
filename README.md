# TwinFocus: Autofocus for Handheld mmWave SAR Imaging via Physical and Digital Twin References
### Authors: [Yadong Li](yadongli.com), [Xinghua Sun](https://xsun2445.github.io/), Qiancheng Li, [Akshay Gadre](https://www.akshaygadre.com/), MobiSys'26.





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
[Download the dataset](https://drive.google.com/file/d/1rsn82nUsJcmIPjcLHiq3hcpd54JDD_MO/view?usp=sharing) and extract it into the following directory:
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

## Citing
If you find this code useful for your research, please consider citing the following paper:
```
@inproceedings{twinfocus,
author = {Li, Yadong and Sun, Xinghua and Li, Qiancheng and Gadre, Akshay},
title = {TwinFocus: Autofocus for Handheld mmWave SAR Imaging via Physical and Digital Twin References},
year = {2026},
isbn = {9798400720277},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3745756.3809185},
doi = {10.1145/3745756.3809185},
booktitle = {Proceedings of the 24th Annual International Conference on Mobile Systems, Applications and Services},
pages = {1–13},
numpages = {13},
location = {University of Cambridge, Cambridge, United Kingdom},
series = {MobiSys '26}
}
```

