import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import cv2

# Configuration
INPUT_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
DATA_PATH = r"C:\Users\harshit\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2"

def prepare_data():
    print("Preparing data...")
    metadata = pd.read_csv(os.path.join(DATA_PATH, 'HAM10000_metadata.csv'))
    
    # Path mapping
    img_dir1 = os.path.join(DATA_PATH, 'HAM10000_images_part_1')
    img_dir2 = os.path.join(DATA_PATH, 'HAM10000_images_part_2')
    
    def get_path(image_id):
        p1 = os.path.join(img_dir1, image_id + '.jpg')
        if os.path.exists(p1): return p1
        return os.path.join(img_dir2, image_id + '.jpg')

    metadata['path'] = metadata['image_id'].apply(get_path)
    
    # Label mapping in alphabetical order (Standard)
    label_map = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
    metadata['label'] = metadata['dx'].map(label_map)
    
    # Split
    train_df, val_df = train_test_split(metadata, test_size=0.15, stratify=metadata['label'], random_state=42)
    
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='path', y_col='dx', # y_col uses string for flow_from_dataframe
        target_size=INPUT_SIZE, class_mode='categorical', batch_size=BATCH_SIZE
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col='path', y_col='dx',
        target_size=INPUT_SIZE, class_mode='categorical', batch_size=BATCH_SIZE
    )
    
    # Calculate Class Weights to handle imbalance
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(train_df['label']), y=train_df['label']
    )
    class_weights = dict(enumerate(weights))
    
    return train_gen, val_gen, class_weights

def build_model():
    print("Building Advanced Transfer Learning Model (MobileNetV2)...")
    base_model = MobileNetV2(input_shape=(*INPUT_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base initially
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    train_gen, val_gen, class_weights = prepare_data()
    model = build_model()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    print("Starting Training Stage 1 (Frozen Base)...")
    model.fit(train_gen, validation_data=val_gen, epochs=10, 
              class_weight=class_weights, callbacks=callbacks)
    
    print("Starting Training Stage 2 (Fine-tuning top layers)...")
    # Unfreeze part of the base model
    model.layers[0].trainable = True
    # Only train layers from 100 onwards
    for layer in model.layers[0].layers[:100]:
        layer.trainable = False
        
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), # Lower LR for fine-tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS - 10, 
                        class_weight=class_weights, callbacks=callbacks)
    
    model.save('optimized_skin_disease_model.h5')
    print("Optimized model saved as 'optimized_skin_disease_model.h5'")
    return history

if __name__ == "__main__":
    train()
