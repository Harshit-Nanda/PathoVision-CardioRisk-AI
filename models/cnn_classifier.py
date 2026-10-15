import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_histopathology_cnn(input_shape=(128, 128, 3), num_classes=3):
    """
    Builds a deep Convolutional Neural Network for microscopic histopathological tissue classification.
    Target Classes:
      0: Normal lung tissue (lung_n)
      1: Lung Adenocarcinoma (lung_aca)
      2: Lung Squamous Cell Carcinoma (lung_scc)
    """
    model = keras.models.Sequential([
        # Block 1
        layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),

        # Classification Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
