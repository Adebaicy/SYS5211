"""
Satellite Image Classification using CNN in TensorFlow

Author: Oluwapelumi Oluwaseyi Adejumo
Date: March 26, 2025
Description:
    This script trains a Convolutional Neural Network (CNN) to classify satellite images 
    into different categories using the dataset from Kaggle.

    The dataset is loaded using `kagglehub`, and image augmentation is applied using 
    `ImageDataGenerator` to improve generalization. The CNN model consists of multiple 
    convolutional and pooling layers, followed by dense layers for classification.

Dependencies:
    - TensorFlow
    - Keras
    - KaggleHub
    - Pandas, NumPy (optional for further analysis)

Usage:
    Run the script directly to train the model.
    ```
    python satellite_classification.py
    ```
    Ensure that the Kaggle API is properly set up before running.

Dataset:
    - The dataset is downloaded from Kaggle (`mahmoudreda55/satellite-image-classification`).
    - It contains multiple classes of satellite images stored in separate directories.

Model Architecture:
    - Conv2D Layers: Feature extraction using 3x3 filters
    - MaxPooling2D: Reducing spatial dimensions
    - Dense Layers: Fully connected layers for classification
    - Dropout: Regularization to prevent overfitting

"""
