# Sparse-ANNs-Training

## Overview
This repository contains Python code for training and evaluating convolutional neural network (CNN), DenseNet, and ResNet models on a small image dataset. 
The code includes functionalities for model pruning, quantization, energy consumption, execution time and memory usage calculation during both training and inference. 
The models are trained on an 8x8, 16x16 and 32x32 pixels image dataset related to malaria detection.

## Dependencies
Make sure you have the following dependencies installed:
<p>&nbsp;</p>
<p>&nbsp;</p>
-Python 3.10 64-bit
<p>&nbsp;</p>
-TensorFlow 2.12 as tf
<p>&nbsp;</p>
-TensorFlow Model Optimization (tfmot) as tfmot
<p>&nbsp;</p>
-from tensorflow import keras
<p>&nbsp;</p>
-NumPy as np
<p>&nbsp;</p>
-Pandas as pd
<p>&nbsp;</p>
-Matplotlib (for plotting results)
<p>&nbsp;</p>
-Psutil
<p>&nbsp;</p>
-Tempfile
<p>&nbsp;</p>
-Tracemalloc
<p>&nbsp;</p>
-os
<p>&nbsp;</p>
-time
<p>&nbsp;</p>
-gc
<p>&nbsp;</p>


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

Directly in the main() function, the user can set up his/her own models (1: CNN, 2: DenseNet, 3: ResNet), as well as sparsity levels and epochs.
After the selection is made, the code will automatically execute untill all the models, sparsity levels and epochs results are saved into the Excel file.
This script trains CNN, DenseNet, and ResNet models with various sparsity levels, quantization options, and epochs. The results, including accuracy, precision, recall, inference time, and energy consumption, are saved to an Excel file.

Here, the gray_scale file are available as well for the users to be used. They function in the same way but the image are processed in black and white.

# Data Preparation
Ensure that your image dataset is organized into the following directory structure:

<img width="211" alt="image" src="https://github.com/MAAntinori/Sparse-ANNs-Training/assets/80471656/a59299e0-02ad-413c-b4f6-6ab10c566e24">
<p>&nbsp;</p>
<img width="118" alt="image" src="https://github.com/MAAntinori/Sparse-ANNs-Training/assets/80471656/0bab0c40-a501-4fae-9e01-cb2071d269e2">
<p>&nbsp;</p>
And then for both training and testing repositories: 
<p>&nbsp;</p>
<img width="82" alt="image" src="https://github.com/MAAntinori/Sparse-ANNs-Training/assets/80471656/9f148749-f1bb-47ea-b85d-b5bb2ff737bb">



# Results
The results of model training and evaluation, including accuracy, precision, recall, inference time, energy consumption, and memory usage, are saved to an Excel file. You can find this file in the "Results" directory with the name: 
test_memory_consumption_model_results_8x8_epochs.xlsx.
It is possible to manually change the saving path as well as the name of the file that will be produced.

# License
This project is licensed under the MIT License.
