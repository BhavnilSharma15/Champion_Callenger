
# Champion-Challenger ML Model with Flask

## 📌 Project Overview

This project demonstrates a simple ML prediction API using Flask.

Two machine learning models are trained and exposed through a REST API:

1. Logistic Regression
2. Random Forest Classifier

The Flask API accepts Age and Estimated Salary as input and returns predictions and prediction probabilities from both models.

## 🏗️ Project Architecture

Client\
   ↓
Flask REST API/n
   ↓
Input Data/
   ↓
├── Logistic Regression/
│
└── Random Forest/
   ↓
Predictions/
   ↓
JSON Response/

# Project Structure

champion-challenger-model/

│
├── app.py
│
├── championTraining.py
├── challengerTraining.py
│
├── Social_Network_Ads.csv
│
├── champion_model.pkl
├── challenger_model.pkl
├── champion_scaler.pkl
└── challenger_scaler.pkl
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── screenshots/
    ├── postman_prediction.png
    └── model_accuracy.png

## 📊 Dataset

Dataset: Social Network Ads

Features:
- Age
- EstimatedSalary

Target:
- Purchased

## 🤖 Models

### Logistic Regression
Used as the baseline/champion model.

### Random Forest Classifier
Used as the challenger model.

## 🚀 How to Run

### 1. Clone repository

git clone <your-repository-url>

### 2. Create virtual environment

python -m venv .venv

### 3. Activate environment

Windows:

.venv\Scripts\activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Run Flask

python app.py

API will run at:

http://127.0.0.1:5000

## 🔌 API Endpoint

POST /predict

### Request

{
    "age": 35,
    "estimated_salary": 80000
}

### Response

{
    "input": {
        "age": 35,
        "estimated_salary": 80000
    },
    "logistic_regression": {
        "prediction": 1,
        "probability": 0.82
    },
    "random_forest": {
        "prediction": 1,
        "probability": 0.91
    }
}

## 📈 Model Evaluation

Add your actual accuracy results here.

| Model | Accuracy |
|---|---|
| Logistic Regression | XX% |
| Random Forest | XX% |

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- Postman
