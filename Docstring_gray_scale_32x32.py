"""
Automated Model Training and Evaluation Script

This script provides functions for creating, training, and evaluating Convolutional Neural Network (CNN),
DenseNet, and ResNet models. It supports experimenting with various model architectures, sparsity levels,
quantization options, and training epochs. The script records key metrics such as accuracy, precision,
recall, inference time, execution time, memory usage, and energy consumption during both training and
inference.
This is the gray_scale version, which means that the the images are transformed into black and white images
and then the code works in the same way. This is done to further compare the precision and accuracy of our models with uncloured images.

Usage:
1. Ensure that TensorFlow, TensorFlow Model Optimization, psutil, numpy, pandas, tracemalloc, and memory_profiler
   are installed.
2. Modify the file paths in the `create_data_generators` function to point to your dataset directories.
3. Run the script by executing `python script_name.py` in the terminal.

Key Functions:
- `create_cnn`: Creates a Convolutional Neural Network (CNN) model.
- `dense_block`: Creates a dense block for a DenseNet model.
- `transition_layer`: Creates a transition layer for a DenseNet model.
- `create_densenet`: Creates a DenseNet model with specified parameters.
- `residual_block`: Creates a residual block for a ResNet model.
- `create_resnet`: Creates a ResNet model.
- `prune_model`: Applies pruning to a given model based on the target sparsity.
- `evaluate_model`: Evaluates a TFLite model using an interpreter and a data generator.
- `create_data_generators`: Updates the data augmentation and loading using `tf.keras.utils.image_dataset_from_directory`.
- `calculate_energy_consumption`: Calculates energy consumption during training and inference.
- `train_and_evaluate_models`: Iterates over different combinations of models, sparsity levels, quantization options,
  and epochs to train and evaluate the models. Results are saved to an Excel file.
- `main`: Main program logic to initiate the training and evaluation process.

Author: Mattia Andrea Antinori
"""
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_model_optimization as tfmot
from memory_profiler import profile
import psutil  # Import the psutil library for getting process memory usage
import tempfile
import numpy as np
import pandas as pd
import tracemalloc
import gc  # Import the garbage collection module to explicitly run garbage collection
import matplotlib. pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Create a Convolutional Neural Network (CNN) model
def create_cnn():
    """
    Creates a Convolutional Neural Network (CNN) model.

    Returns:
    - model: A Keras CNN model.
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(32, 32, 1)),
        keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create a dense block for a DenseNet model
def dense_block(x, num_layers, growth_rate):
    """
    Creates a dense block for a DenseNet model.

    Args:
    - x: Input tensor.
    - num_layers: Number of layers in the dense block.
    - growth_rate: Growth rate for convolutional layers.

    Returns:
    - x: Output tensor after passing through the dense block.
    """
    # Implementation of the dense block function as before
    for i in range(num_layers):
        # BN - ReLU - Conv(1x1) - BN - ReLU - Conv(3x3)
        # Within the dense block, Batch Normalization, 
        # ReLU activation, and 1x1 and 3x3 convolutions 
        # are applied in an iterative manner
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(growth_rate, (1, 1), padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(growth_rate, (3, 3), padding='same')(x1)
        # Concatenate the output with the input
        x = Concatenate()([x, x1])    
    return x

# Create a transition layer for a DenseNet model
def transition_layer(x, compression_factor):
    """
    Creates a transition layer for a DenseNet model.

    Args:
    - x: Input tensor.
    - compression_factor: Factor to reduce the number of filters.

    Returns:
    - x: Output tensor after passing through the transition layer.
    """
    # Implementation of the transition layer function as before
     # BN - Conv(1x1) - AveragePooling(2x2)
    #used to reduce the number of filters while maintaining spatial dimensions
    x = BatchNormalization()(x)
    x = Conv2D(int(tf.keras.backend.int_shape(x)[-1] * compression_factor), (1, 1), padding='same')(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

# Create a DenseNet model
def create_densenet(input_shape, num_classes, num_layers_per_block, growth_rate, compression_factor):
    """
    Creates a DenseNet model with specified parameters.

    Args:
    - input_shape: Shape of the input tensor.
    - num_classes: Number of output classes.
    - num_layers_per_block: List specifying the number of layers in each dense block.
    - growth_rate: Growth rate for convolutional layers.
    - compression_factor: Factor to reduce the number of filters in transition layers.

    Returns:
    - model: A Keras DenseNet model.
    """
    # Implementation of the DenseNet model function as before
     # Input layer
    inputs = Input(shape=input_shape)
    
    # Conv(3x3) with 16 filters, stride=1
    x = Conv2D(16, (3, 3), padding='same', strides=(1, 1))(inputs)
    
    # Dense blocks with transition layers
    for i, num_layers in enumerate(num_layers_per_block):
        x = dense_block(x, num_layers, growth_rate)
        if i != len(num_layers_per_block) - 1:
            x = transition_layer(x, compression_factor)
    
    # Global average pooling and fully connected layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create a residual block for a ResNet model
def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1), activation='relu'):
    """
    Creates a residual block for a ResNet model.

    Args:
    - x: Input tensor.
    - filters: Number of filters in the convolutional layers.
    - kernel_size: Size of the convolutional kernel.
    - strides: Stride for the convolutional layers.
    - activation: Activation function.

    Returns:
    - y: Output tensor after passing through the residual block.
    """
    # Implementation of the residual block function as before
    y = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)
    y = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    if strides != (1, 1) or x.shape[-1] != filters:
        x = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)

    y = add([x, y])
    y = Activation(activation)(y)
    return y


# Create a ResNet model
def create_resnet(input_shape, num_classes):
    """
    Creates a ResNet model.

    Args:
    - input_shape: Shape of the input tensor.
    - num_classes: Number of output classes.

    Returns:
    - model: A Keras ResNet model.
    """
    # Implementation of the ResNet model function as before
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block(x, filters=8, strides=(2, 2))
    x = residual_block(x, filters=8, strides=(1, 1))
    x = residual_block(x, filters=8, strides=(1, 1))

    x = residual_block(x, filters=16, strides=(2, 2))
    x = residual_block(x, filters=16, strides=(1, 1))
    x = residual_block(x, filters=16, strides=(1, 1))

    x = residual_block(x, filters=32, strides=(2, 2))
    x = residual_block(x, filters=32, strides=(1, 1))
    x = residual_block(x, filters=32, strides=(1, 1))

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def prune_model(model, target_sparsity):
    """
    Applies pruning to a given model based on the target sparsity.

    Args:
    - model: Keras model to be pruned.
    - target_sparsity: Target sparsity level.

    Returns:
    - model_pruned: Pruned Keras model.
    """
    # Implementation of the prune_model function as before
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=target_sparsity,
            begin_step=0,
            end_step=-1,
            frequency=1
        )
    }

    model_pruned = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_pruned.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model_pruned


# Create a function to visualize images from a dataset
def visualize_images(dataset, num_images=5):
    """
    Visualizes images from a given dataset.

    Args:
    - dataset: TensorFlow dataset containing images and labels.
    - num_images: Number of images to visualize.

    Returns:
    - None
    """
    plt.figure(figsize=(15, 3))
    for i, (images, labels) in enumerate(dataset.take(num_images)):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[0].numpy().squeeze(), cmap='gray')  # Display the first image in the batch
        plt.title(f"Label: {labels[0].numpy()}")
        plt.axis('off')
    plt.show()



# Evaluate a TFLite model using an interpreter and a data generator
def evaluate_model(interpreter, data_generator):
    """
    Evaluates a TFLite model using an interpreter and a data generator.

    Args:
    - interpreter: TFLite interpreter for the model.
    - data_generator: Data generator for the evaluation dataset.

    Returns:
    - accuracy: Model accuracy.
    - precision: Model precision.
    - recall: Model recall.
    - inference_time_per_frame: Average inference time per frame.
    """
    # Implementation of the evaluate_model function as before
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image generated by the data generator.
    prediction_probs = []
    test_labels = []
    inference_times = []
    for batch_images, batch_labels in data_generator:
        batch_size = batch_images.shape[0]

        # Pre-processing: convert to float32 to match with the model's input data format.
        batch_images = tf.cast(batch_images, tf.float32)

        # Run inference on each image in the batch and measure the inference time.
        for j in range(batch_size):
            image = batch_images[j:j+1]
            start_time = time.perf_counter()
            interpreter.set_tensor(input_index, image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            probs = output.squeeze()
            prediction_probs.append(probs)
            test_labels.append(batch_labels[j])

    # Convert probabilities to binary predictions.
    prediction_probs = np.array(prediction_probs)
    prediction_binary = (prediction_probs > 0.5).astype(int)

    # Calculate accuracy, precision, and recall.
    test_labels = np.array(test_labels)
    confusion_matrix = tf.math.confusion_matrix(test_labels, prediction_binary)
    true_positives = confusion_matrix[1][1]
    false_positives = confusion_matrix[0][1]
    false_negatives = confusion_matrix[1][0]
    accuracy = (prediction_binary == test_labels).mean()
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Calculate the average inference time per frame.
    inference_time_per_frame = sum(inference_times) / len(inference_times)

    return accuracy, precision.numpy(), recall.numpy(), inference_time_per_frame

# Update the data augmentation and loading using tf.keras.utils.image_dataset_from_directory
def create_data_generators(batch_size=16):
    """
    Updates the data augmentation and loading using `tf.keras.utils.image_dataset_from_directory`.

    Args:
    - batch_size: Batch size for the data generators.

    Returns:
    - train_data: Training data generator.
    - val_data: Validation data generator.
    - test_data: Testing data generator.
    """
    train_dir = r'C:\ANNs\Data\Malaria dataset split - training and testing\Testing_data_cells'
    test_dir = r'C:\ANNs\Data\Malaria dataset split - training and testing\Training_data_cells'

    train_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(32, 32),
        color_mode="grayscale",
        batch_size=batch_size,
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(32, 32),
        color_mode="grayscale",
        batch_size=batch_size,
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(32, 32),
        color_mode="grayscale",
        batch_size=batch_size,
    )

    # Apply data augmentation
    data_augmentation = tf.keras.Sequential([
        preprocessing.Rescaling(1./255),
        preprocessing.RandomFlip("horizontal"),
       # preprocessing.RandomFlip("vertical"),
    ])

    train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))

    return train_data, val_data, test_data

def calculate_energy_consumption(model, num_epochs, inference_time, power_consumption):
    """
    Calculates energy consumption during training and inference.

    Args:
    - model: Keras model.
    - num_epochs: Number of training epochs.
    - inference_time: Average inference time per frame during evaluation.
    - power_consumption: Power consumption in watts.

    Returns:
    - training_energy: Energy consumption during training (watt-hours).
    - inference_energy: Energy consumption during inference (watt-hours).
    """
    # Assuming power consumption is given in watts
    # and num_epochs is the number of training epochs

    # Calculate energy consumption during training
    training_time = num_epochs * inference_time  # Assuming each epoch takes the same time as one inference
    training_energy = training_time * power_consumption  # Energy in watt-hours

    # Assuming power consumption during inference is constant
    inference_energy = inference_time * power_consumption  # Energy in watt-hours

    return training_energy, inference_energy
# Create a new function for automated model training and evaluation
def train_and_evaluate_models(models,sparsity_levels,epochs_to_try):
    """
    Iterates over different combinations of models, sparsity levels, quantization options,
    and epochs to train and evaluate the models. Results are saved to an Excel file.

    Args:
    - models: List of model choices (1: CNN, 2: DenseNet, 3: ResNet).
    - sparsity_levels: List of sparsity levels to experiment with.
    - epochs_to_try: List of epoch values to iterate through.

    Returns:
    - results_list: List containing detailed results for each combination.
    """
    # Create an empty DataFrame to store the results
    results_list = []
    results_df = pd.DataFrame()
    # Models and sparsity levels to iterate over
    models = ['1', '2','3']  # Model choices (1: CNN, 2: DenseNet, 3: ResNet)
    sparsity_levels = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95]
# Create a list of epoch values to iterate through
    epochs_to_try = [0,2, 5,10, 15,20,25,30]
    #epochs_to_try = [2, 5,10]
    
    start_time = time.time()
    
    # Assuming power consumption during training and inference
    power_consumption_training = 150  # Adjust this value based on your hardware during training
    power_consumption_inference = 50  # Adjust this value based on your hardware during inference


# Loop over models, sparsity levels, and epochs
    for model_choice in models:
        for sparsity in sparsity_levels:
            for quantize_model in [True, False]:
                for num_epochs in epochs_to_try:
                    tracemalloc.start()
                    gc.collect()
                    # Clear the current memory allocation traces
                    tracemalloc.clear_traces()

                    # Create the selected model
                    if model_choice == '1':
                        model_name = "CNN"
                        model = create_cnn()
                    elif model_choice == '2':
                        model_name = "DenseNet"
                        model = create_densenet(input_shape=(32, 32, 1), num_classes=1, num_layers_per_block=[4, 4, 4], growth_rate=12, compression_factor=0.5)
                    elif model_choice == '3':
                        model_name = "ResNet"
                        model = create_resnet(input_shape=(32, 32, 1), num_classes=1)
                    else:
                        print("Invalid choice. Please choose 1, 2, or 3.")
                        continue

                    quantization_info = "Quantized" if quantize_model else "Unquantized"
                    
                      
                    
                    # Print the model, sparsity, and epoch being trained and evaluated
                    print(f"Model: {model_name}, Sparsity: {sparsity}, Quantization: {quantization_info}, Epochs: {num_epochs}")

                    # Compile the model
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    
                    early_stopping_callback = EarlyStopping(
                        monitor='accuracy',  # You can change this to another metric like 'val_accuracy'
                        patience=100,           # Number of epochs with no improvement after which training will be stopped
                        restore_best_weights=True
                    )

                    # Usage
                    train_data, val_data, test_data = create_data_generators()

                    # Train the model with the desired sparsity and number of epochs
                    if sparsity == 0:
                    # No pruning callbacks needed
                        model.fit(
                            train_data,
                            steps_per_epoch=len(train_data),
                            epochs=num_epochs,
                            validation_data=val_data,
                            validation_steps=len(val_data),
                            callbacks=[early_stopping_callback]
                    )

                    else:
                        # Prune the model and train
                        model_pruned = prune_model(model, target_sparsity=sparsity)
                        pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
                        callbacks = [pruning_callback, early_stopping_callback]

                        model_pruned.fit(
                            train_data,
                            steps_per_epoch=len(train_data),
                            epochs=num_epochs,
                            validation_data=val_data,
                            validation_steps=len(val_data),
                            callbacks=callbacks
                        )

                        model = tfmot.sparsity.keras.strip_pruning(model_pruned)

                    # Record the end time
                    end_time = time.time()

                    # Calculate the execution time
                    execution_time = end_time - start_time

                    # Print the execution time
                    print(f"Total execution time: {execution_time} seconds")

                    # Save the execution time to the DataFrame
                    results_df['Execution Time'] = execution_time

                    current, peak = tracemalloc.get_traced_memory()
                    print(f"Current memory usage: {current / 10**6} MB, Peak memory usage: {peak / 10**6} MB")

                    tracemalloc.stop()
                      
                     # Convert the model to TensorFlow Lite
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    tflite_model = converter.convert()

                    _, tflite_file = tempfile.mkstemp('.tflite')

                    with open(tflite_file, 'wb') as f:
                        f.write(tflite_model)

                    print('Saved TFLite model to:', tflite_file)

                    # Convert to quantized TFLite
                    if quantize_model:
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        tflite_quantized_model = converter.convert()

                        _, quantized_file = tempfile.mkstemp('.tflite')

                        with open(quantized_file, 'wb') as f:
                            f.write(tflite_quantized_model)

                        print('Saved quantized TFLite model to:', quantized_file)

               
                    interpreter = tf.lite.Interpreter(model_content=tflite_model)
                    interpreter.allocate_tensors()

                    accuracy, precision, recall, inference_time = evaluate_model(interpreter, test_data)
                    
                    # Assuming power consumption during training and inference
                    power_consumption = power_consumption_training if num_epochs > 0 else power_consumption_inference

                   
                    # Calculate energy consumption
                    training_energy, inference_energy = calculate_energy_consumption(model, num_epochs, inference_time, power_consumption)
                    
                    
                    # Print or save the energy consumption values as needed
                    print("Training Energy Consumption:", training_energy, "Wh")
                    print("Inference Energy Consumption:", inference_energy, "Wh")
                    print("Model:", model_choice)
                    print("Sparsity:", sparsity)
                    print("Quantized:", quantize_model)
                    print("Epochs:", num_epochs)
                    print("Accuracy:", accuracy)
                    print("Precision:", precision)
                    print("Recall:", recall)
                    print("Inference Time per Frame:", inference_time)

                    # Save the results to the DataFrame
                    result= {
                            'Model': model_choice,
                            'Sparsity': sparsity,
                            'Quantized': quantize_model,
                            'Epochs': num_epochs,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'Inference Time': inference_time,
                            'Execution Time in seconds': execution_time,
                            'Inference Energy Consumption in Watts': inference_energy,
                            'Memory Usage in MB' : training_energy 
                        }
                    results_list.append(result)
                    results_df = pd.DataFrame(results_list)
                                # Define the Excel file name
                    file_name = 'test_energy_memory_consumption_model__gray_scale_results_32x32_epochs.xlsx'

                                # Define the file path
                    file_path = os.path.join(r'C:\ANNs\Results', file_name)

                                # Save the DataFrame to an Excel file
                    results_df.to_excel(file_path, index=False)
    return results_list
     
# Main program logic
def main():
    """
    Main program logic to initiate the training and evaluation process.

    Returns:
    - None
    """
    train_data, _, _ = create_data_generators()
     # Models and sparsity levels to iterate over
    models = ['1', '2','3']  # Model choices (1: CNN, 2: DenseNet, 3: ResNet)
    sparsity_levels = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95]
    # Create a list of epoch values to iterate through
    epochs_to_try = [0,2, 5,10, 15,20,25,30]
    #epochs_to_try = [2, 5,10]
    train_and_evaluate_models(models,sparsity_levels,epochs_to_try)
       
if __name__ == '__main__':
        main()
