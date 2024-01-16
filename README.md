# Sparse-ANNs-Training

## Overview
This repository contains Python code for training and evaluating convolutional neural network (CNN), DenseNet, and ResNet models on a small image dataset. 
The code includes functionalities for model pruning, quantization, energy consumption, execution time and memory usage calculation during both training and inference. 
The models are trained on an 8x8, 16x16 and 32x32 pixels image dataset related to malaria detection.

## Dependencies
Make sure you have the following dependencies installed:
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
from reportlab.lib.pagesizes import letter
<p>&nbsp;</p>
from reportlab.pdfgen import canvas


You can install these dependencies using the following command:
pip install (dependency)

## Installation
Clone the repository:
git clone https://github.com/your_username/your_repository.git
<p>&nbsp;</p>
cd your_repository

Install the dependencies (see Dependencies).

# Data Preparation
Ensure that your image dataset is organized into the following directory structure:

<img width="211" alt="image" src="https://github.com/MAAntinori/Sparse-ANNs-Training/assets/80471656/a59299e0-02ad-413c-b4f6-6ab10c566e24">
<p>&nbsp;</p>
<img width="118" alt="image" src="https://github.com/MAAntinori/Sparse-ANNs-Training/assets/80471656/0bab0c40-a501-4fae-9e01-cb2071d269e2">
<p>&nbsp;</p>
And then for both training and testing repositories: 
<p>&nbsp;</p>
<img width="82" alt="image" src="https://github.com/MAAntinori/Sparse-ANNs-Training/assets/80471656/9f148749-f1bb-47ea-b85d-b5bb2ff737bb">

## Usage Training and Evaluation
To train and evaluate models, run the main() function in the Montecarlo.py script:

Directly in the main() function, the user can set up his/her own models (1: CNN, 2: DenseNet, 3: ResNet), as well as sparsity levels, epochs and number of iterations (Monte Carlo simulation).
<p>&nbsp;</p>
The function monte_carlo_simulation() takes as arguments iterations, models, sparsity_levels, and epochs_to_try. Once the number of iterations is defined directly in the main() function, the code will start, and it will produce both the results in the Excel file and the mean of the results obtained by looping over the number of iterations selected by the user.
<p>&nbsp;</p>
Not only that, the model will ask the user to select the 'rgb'mode (images are coloures) or to use the 'grayscale' mode, where the images are in black and white.
After the user has prompted the color mode, he/she will be asked to insert the desired image resolution, 8x8, 16x16 or 32x32.
Finally, the code will automatically execute untill all the models, sparsity levels and epochs results are saved into the Excel file and a PDF report, with the means of the results, is created .
This script trains CNN, DenseNet, and ResNet models with various sparsity levels, quantization options, and epochs. The results, including accuracy, precision, recall, inference time, and energy consumption, are saved to an Excel file and into the PDF file.


# Results
The results of model training and evaluation, including accuracy, precision, recall, inference time, energy consumption, and memory usage, are saved to an Excel file. You can find this file in the "Results" directory with the name: 
test_memory_consumption_model_results_8x8_epochs.xlsx.
It is possible to manually change the saving path as well as the name of the file that will be produced.

# License
This project is licensed under the MIT License.
