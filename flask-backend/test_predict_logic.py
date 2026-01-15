try:
    import tf_keras as keras
    from tf_keras.models import load_model
except ImportError:
    try:
        from tensorflow import keras
        from tensorflow.keras.models import load_model
    except ImportError:
        import keras
        from keras.models import load_model

import cv2
import numpy as np
import os

# Use the model main.py uses
MODEL_PATH = 'my_skin_disease_pred_model.h5'


TEST_IMAGE = '../1.jpeg'


class_names = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions'
}

def test_logic():
    print(f"Loading model: {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Loading image: {TEST_IMAGE}...")
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print("Failed to load image.")
        return

    try:
        # Preprocess image (resize and normalize) exactly as in main.py
        print("Processing image...")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)


        print(f"Input shape: {image.shape}")
        
        print("Running prediction...")
        predictions = model.predict(image)
        print(f"Raw predictions: {predictions}")

        predicted_class_index = np.argmax(predictions[0])
        predicted_prob = predictions[0][predicted_class_index]
        predicted_class_name = class_names[predicted_class_index]

        print(f"Result: {predicted_class_name} with probability {predicted_prob}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logic()
