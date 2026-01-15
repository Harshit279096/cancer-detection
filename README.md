# Skin Cancer Detection using Deep CNN

![Home Page Screenshot](screenshots/ss1.png)
![Take a test Screenshot](screenshots/ss2.png)

This project is a full-stack web application designed for skin cancer detection. It combines a React-based frontend, a Node.js/Express management backend, and a specialized Flask backend for high-accuracy AI prediction.

## 🏗️ Architecture Overview

The system consists of three main components:
1.  **Frontend**: Built with React, Vite, and Tailwind CSS.
2.  **Management Backend (Node.js)**: Handles users, authentication, database (MongoDB), and result history.
3.  **Research/AI Backend (Flask)**: Hosts the TensorFlow/Keras model for actual image classification.

---

## 🛠️ Installation & Setup

Before starting, ensure you have **Node.js (v18+)**, **Python (3.12+)**, and **MongoDB** installed.

### 1. Flask Research/AI Backend
This must be running for predictions to work.
```bash
cd flask-backend
# Windows:
.\venv\Scripts\activate
python main.py
# (Or simply run run_backend.bat)
```

### 2. Management Backend (Node.js)
```bash
cd backend
npm install
npm start
```
*Note: Ensure your MongoDB server is running.*

### 3. Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

---

## ✨ Features

- **User Authentication**: Secure signup and login system.
- **AI Prediction**: Upload skin lesion images for immediate classification.
- **Result Management**: Save and view history of previous tests.
- **Reporting**: Ability to share or export results for medical consultation.

---

## 📁 Repository Structure

- `/frontend`: React application source code.
- `/backend`: Node.js Express server for user/data management.
- `/flask-backend`: Python Flask server for AI model inference.
- `/screenshots`: UI images and design diagrams.

---

## 📊 Visual Aid

![High Level Design](screenshots/diagram.png)
![Test Results 1](screenshots/ss4.png)
![Test Results 2](screenshots/ss3.png)
