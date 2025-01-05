# Struct-X: Enhancing the Reasoning Capabilities of Large Language Models in Structured Data Scenarios

![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-orange)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains the official implementation of the Struct-X framework as described in our paper: "Struct-X: Enhancing the Reasoning Capabilities of Large Language Models in Structured Data Scenarios" (ACMKDD 2025).

## Overview

Struct-X is a novel framework that enhances knowledge graph question answering through structural learning and hierarchical relationship modeling. Our approach achieves state-of-the-art performance on multiple QA benchmarks while maintaining computational efficiency.

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/struct-x.git
cd struct-x

# Create and activate conda environment
conda create -n structx python=3.8
conda activate structx

# Install requirements
pip install -r requirements.txt
```

## Directory Structure

```
structx/
├── auxiliary/           # Auxiliary functions and utilities
├── injection/          # Knowledge injection modules
├── kg/                 # Knowledge graph processing
├── retrieval/          # Information retrieval components
├── topologyencoder/    # Topology encoding implementation
├── structx_train.py    # Main training script
└── finetune13b.sh     # Fine-tuning script for 13B model
```

## Usage

### Data Preparation

1. Download the required datasets:
```bash
# Download BioASQ dataset
python scripts/download_bioasq.py

# Download MedQA dataset
python scripts/download_medqa.py
```

2. Preprocess the knowledge graph:
```bash
python kg/process_kg.py --input_path data/raw_kg --output_path data/processed_kg
```

### Training

To train the model from scratch:

```bash
python structx_train.py \
    --data_dir data/processed \
    --model_type base \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --save_dir checkpoints/
```

For fine-tuning the 13B model:

```bash
bash finetune13b.sh
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data data/processed/test.json \
    --output_dir results/
```

## Main Results

Our model achieves the following performance on BioASQ and MedQA tasks:

| Task    | Accuracy | F1 Score |
|---------|----------|-----------|
| BioASQ  | 79.8%    | 79.1%     |
| MedQA   | 77.5%    | 76.7%     |

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{tan2025struct,
  author = {Tan, Xiaoyu and Wang, Haoyu and Qiu, Xihe and Cheng, Leijun and Cheng, Yuan and Chu, Wei and Xu, Yinghui and Qi, Yuan},
  title = {Struct-X: Enhancing the Reasoning Capabilities of Large Language Models in Structured Data Scenarios},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  series = {KDD '25},
  year = {2025},
  month = aug,
  location = {Toronto, ON, Canada},
  pages = {15},
  numpages = {15},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3690624.3709381},
  volume = {1}
}
```

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- transformers 4.5.0+
- numpy 1.19.2+
- scipy 1.7.0+
- scikit-learn 0.24.2+


## Contact

For questions and feedback, please open an issue in this repository or contact [ambityuki@gmail.com].
