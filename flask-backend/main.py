import os
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
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the working model
model = load_model('my_skin_disease_pred_model.h5')

# Define class names (Alphabetical order - matches .h5 model)
class_names = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions'
}

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Flask backend is running'})


@app.route('/predict', methods=['POST'])
def predict():
    print("Received request at /predict")
    print(f"Files in request: {request.files}")
    try:
        if request.method == 'POST':
            if 'image' not in request.files:
                print("No image part in request")
                return jsonify({'error': 'No image part'}), 400
            file = request.files['image']
            predicted_class_index, predicted_class_name, predicted_prob, image = predict_image(file)

            res = {
                'predicted_class': predicted_class_name,
                'prediction_probability': str(predicted_prob)
            }
            print(f"Prediction result: {predicted_class_name} ({predicted_prob})")
            return jsonify(res)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def predict_image(file):
    # Read image from Flask file object
    try:
        file_bytes = file.read()
        if not file_bytes:
            raise ValueError("File is empty or could not be read")
        
        file.seek(0)  # Reset file pointer for potential reuse
        
        # Convert bytes to numpy array
        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Could not decode image. File size: {len(file_bytes)} bytes. Please ensure the file is a valid image.")
    except AttributeError as e:
        # If file.read() doesn't work, try alternative method
        file_bytes = file.stream.read()
        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image using stream method.")

    # Preprocess image (resize and normalize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Fix: Model expects RGB
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0


    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)

    predicted_class_index = np.argmax(predictions[0])
    predicted_prob = predictions[0][predicted_class_index]
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_index, predicted_class_name, predicted_prob, image[0]




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)
