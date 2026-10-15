import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from models.cnn_classifier import build_histopathology_cnn
from data.generate_demo_data import generate_synthetic_histology_data

data_path = r'data\lung_subset_demo'
if not os.path.exists(data_path):
    print("Generating synthetic histopathology dataset for demonstration...")
    generate_synthetic_histology_data(data_path, n_per_class=30)

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 5

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

model = build_histopathology_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=3)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nStarting training of Histopathology CNN...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

print("\nTraining completed successfully.")
