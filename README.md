# Sparse-ANNs-Training

## Overview
This repository contains Python code for training and evaluating convolutional neural network (CNN), DenseNet, and ResNet models on a small image dataset. 
The code includes functionalities for model pruning, quantization, energy consumption, execution time and memory usage calculation during both training and inference. 
The models are trained on an 8x8, 16x16 and 32x32 pixels image dataset related to malaria detection.

## Dependencies
Make sure you have the following dependencies installed:

-Python 3.10 64-bit
-TensorFlow 2.12 as tf
-TensorFlow Model Optimization (tfmot) as tfmot
-from tensorflow import keras
-NumPy as np
-Pandas as pd
-Matplotlib (for plotting results)
-Psutil
-Tempfile
-Tracemalloc
-os
-time
-gc


You can install these dependencies using the following command:
pip install (dependency)

## Installation
Clone the repository:
git clone https://github.com/your_username/your_repository.git
cd your_repository

Install the dependencies (see Dependencies).

## Usage Training and Evaluation
To train and evaluate models, run the main() function in the model_evaluation.py script:

python model_evaluation.py

This script trains CNN, DenseNet, and ResNet models with various sparsity levels, quantization options, and epochs. The results, including accuracy, precision, recall, inference time, and energy consumption, are saved to an Excel file.


# Results
The results of model training and evaluation, including accuracy, precision, recall, inference time, energy consumption, and memory usage, are saved to an Excel file. You can find this file in the "Results" directory with the name: 
test_memory_consumption_model_results_8x8_epochs.xlsx.
It is possible to manually change the saving path as well as the name of the file that will be produced.

# License
This project is licensed under the MIT License.
