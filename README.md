
# Attention is All You Need - Reimplementation

This repository contains a reimplementation of the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. The original model introduced the Transformer architecture, which has since become the foundation for many state-of-the-art models in natural language processing.

## Overview

In this project, I have reimplemented the Transformer model following a tutorial and adapted it to work with a different dataset sourced from Hugging Face. This repository includes all necessary code, configuration files, scripts to train and evaluate the model, and a GUI for easy use of the translator.

## Dataset

The dataset used in this reimplementation is obtained from [Hugging Face Datasets](https://huggingface.co/datasets). For this project, I used the `opus-100`, `ru-en` dataset, which can be found [here](https://huggingface.co/datasets/Helsinki-NLP/opus-100).

## Repository Structure

- `src/`: Contains the source code for the Transformer model.
  - `model.py`: Defines the Transformer model architecture.
  - `train.py`: Script for training the model.
  - `dataset.py`: Script for loading and preprocessing the dataset.
  - `app.py`: Script for running the GUI translator application.
- `config`: Configuration file for different training and evaluation setups.
- `results/`: Directory to store training logs, model checkpoints, and evaluation results.

## Installation

To run the code in this repository, you'll need to have Python installed along with several dependencies. You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```
## Usage

### Training the Model

To train the Transformer model, run the following command:

```bash
python src/train.py
```

### Loading and Preprocessing the Data

To preprocess the dataset, run:

```bash
python data/dataset.py
```

### Using the GUI Translator

To use the GUI for translation, run:

```bash
python src/app.py
```

## Configuration

The config files contain hyperparameters and other settings for training and evaluation. You can modify these parameters to experiment with different configurations.

## Results

Results of the training and evaluation, including metrics and model checkpoints, are stored in the `results/` directory. Example results can be found in the `results/` subdirectory.

## Limitations

Due to computational constraints, I used a small dataset for training the model. As a result, the translation performance may not be perfect and might not generalize well to unseen data. I encourage users to experiment with larger datasets and more extensive training if resources permit.

## Acknowledgements

This project was inspired by the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) and follows a tutorial for implementation. Special thanks to the authors of the original paper and the creators of the Hugging Face Datasets library.

