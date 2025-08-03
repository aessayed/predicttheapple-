from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open("mltrainig.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("indexapple.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input features from the form
    float_features = [float(x) for x in request.form.values()]

    # Check if the input features are valid
    if len(float_features) != 7:
        return render_template("indexapple.html", prediction_text="Invalid input features")

    # Convert the input features to a numpy array
    features = np.array(float_features).reshape(1, 7)

    # Make a prediction using the model
    prediction = model.predict(features)

    # Check if the prediction is good or not
    if prediction == 0:
        prediction_text = "The apple is not good"
    else:
        prediction_text = "The apple is good"

    return render_template("indexapple.html", prediction_text=prediction_text)

if __name__ == "__main__":
    # Test the model with some sample input features
    sample_features = np.array([[-0.292023862, -1.351281995 , -1.738429162 , -0.342615928, 2.838635512 , -0.038033328, 2.621636473]]).reshape(1, 7)
    prediction = model.predict(sample_features)
    if prediction == 0:
        prediction_text = "The apple is not good"
    else:
        prediction_text = "The apple is good"
    print(prediction_text)
    app.run(debug=True)