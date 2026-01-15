# Skin Cancer Detection - Flask Backend

This is the Flask backend server for the Skin Cancer Detection application. it uses a deep learning model to predict the type of skin lesion from an uploaded image.

## 🚀 Quick Start (Windows)

If you are on Windows, you can start the backend immediately by double-clicking:
👉 **`run_backend.bat`**

This will automatically use the project's virtual environment and start the server on `http://127.0.0.1:5000`.

---

## 🛠️ Manual Setup

If you need to set up the environment from scratch, follow these steps:

### 1. Prerequisites
- **Python 3.12+** installed.
- (Optional but recommended) **Git** for version control.

### 2. Create Virtual Environment
Open your terminal in the `flask-backend` directory and run:
```powershell
python -m venv venv
```

### 3. Activate and Install Dependencies
```powershell
# Activate venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 4. Run the Server
```powershell
python main.py
```

---

## 📡 API Documentation

The server runs on **`http://127.0.0.1:5000`**.

### 1. Health Check
- **URL:** `/`
- **Method:** `GET`
- **Description:** Checks if the backend is running.

### 2. Predict Image
- **URL:** `/predict`
- **Method:** `POST`
- **Body:** `multipart/form-data`
- **Fields:**
  - `image`: (File) The image of the skin lesion to analyze.
- **Success Response:**
  ```json
  {
    "predicted_class": "Melanocytic nevi",
    "prediction_probability": "0.9998"
  }
  ```

---

## 📁 Project Structure

- `main.py`: The entry point for the Flask application and prediction logic.
- `my_skin_disease_pred_model.h5`: The trained deep learning model.
- `requirements.txt`: List of Python dependencies.
- `run_backend.bat`: Shortcut script to start the backend.
- `venv/`: The virtual environment containing isolated dependencies.
- `archive/`: Old models, logs, and experimental scripts (safe to ignore).

## ⚠️ Troubleshooting

### DLL Load Failed
If you see an error like `ImportError: DLL load failed while importing _pywrap_tensorflow_internal`, it means your base Python environment is conflicting with TensorFlow. 
**Solution:** Always run the project inside the `venv` as described in the setup steps.