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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

# Download Dataset from Kaggle
path = kagglehub.dataset_download("mahmoudreda55/satellite-image-classification")
path = path + '/' + 'data'  # Ensure correct dataset path
print("Path to dataset files:", path)

# Set Random Seed for Reproducibility
tf.random.set_seed(42)

# Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values between 0 and 1
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True,     # Horizontal flipping
    rotation_range=20,        # Random rotation
    validation_split=0.2      # Splitting dataset (80% train, 20% validation)
)

# Load Training Dataset
train_ds = train_datagen.flow_from_directory(
    path,
    target_size=(150, 150),   # Resize images to 150x150 pixels
    batch_size=4,             # Small batch size for training
    class_mode="sparse",      # Use sparse labels (integer class labels)
    subset="training",        # Training subset
    seed=123                  # Seed for consistent data shuffling
)

# Load Validation Dataset (Only Rescaling)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_ds = val_datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=32,            # Larger batch size for validation
    class_mode="sparse",      # Sparse labels
    subset="validation",      # Validation subset
    seed=123
)

# Define CNN Model Architecture
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),  # First Conv Layer
    layers.MaxPooling2D(2,2),  # Pooling Layer
    
    layers.Conv2D(64, (3,3), activation='relu'),  # Second Conv Layer
    layers.MaxPooling2D(2,2),  # Pooling Layer
    
    layers.Conv2D(128, (3,3), activation='relu'),  # Third Conv Layer
    layers.MaxPooling2D(2,2),  # Pooling Layer

    layers.Flatten(),          # Flatten feature maps to 1D
    layers.Dense(128, activation='relu'),  # Fully Connected Layer
    layers.Dropout(0.5),       # Dropout to prevent overfitting
    layers.Dense(4, activation='softmax')  # Output Layer (4 classes, using softmax activation)
])

# Print Model Summary
model.summary()

# Compile the Model
model.compile(optimizer='adam',                          # Adam optimizer
              loss='sparse_categorical_crossentropy',    # Loss function for multi-class classification
              metrics=['accuracy'])                      # Track accuracy metric

# Train the Model
history = model.fit(
    train_ds, 
    epochs=5,                 # Number of training epochs
    validation_data=val_ds     # Validation dataset
)


