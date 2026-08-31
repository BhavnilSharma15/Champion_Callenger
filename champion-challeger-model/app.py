from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

#losd Logistic regression  model

logistic_model = joblib.load('champion.pkl')
logistic_scaler=joblib.load('champion_scaler.pkl')

#Load Random Forest model

random_forest_model = joblib.load("challenger.pkl")
random_forest_scaler = joblib.load("challenger_scaler.pkl")


@app.route("/")
def home():
    return "ML API is running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    age = data["age"]
    salary = data["estimated_salary"]

    # Create input
    input_data = np.array([[age, salary]])


    # -------------------------------
    # Logistic Regression Prediction
    # -------------------------------

    logistic_input = logistic_scaler.transform(input_data)

    logistic_prediction = logistic_model.predict(
        logistic_input
    )[0]

    logistic_probability = logistic_model.predict_proba(
        logistic_input
    )[0][1]


    # -------------------------------
    # Random Forest Prediction
    # -------------------------------

    random_forest_input = random_forest_scaler.transform(
        input_data
    )

    random_forest_prediction = random_forest_model.predict(
        random_forest_input
    )[0]

    random_forest_probability = random_forest_model.predict_proba(
        random_forest_input
    )[0][1]


    # Return both predictions

    return jsonify({

        "input": {
            "age": age,
            "estimated_salary": salary
        },

        "logistic_regression": {
            "prediction": int(logistic_prediction),
            "probability": float(logistic_probability)
        },

        "random_forest": {
            "prediction": int(random_forest_prediction),
            "probability": float(random_forest_probability)
        }

    })


if __name__ == "__main__":
    app.run(debug=True)
