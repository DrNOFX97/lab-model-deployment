import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

# Determine the base directory dynamically
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "../ufo-model.pkl")
data_path = os.path.join(base_dir, "../data/ufos.csv")

# Load the model
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]
    return render_template("index.html", prediction_text=f"Likely country: {countries[output]}")

@app.route("/train", methods=["POST"])
def train():
    ufos = pd.read_csv(data_path)
    ufos = pd.DataFrame({
        'Seconds': ufos['duration (seconds)'],
        'Country': ufos['country'],
        'Latitude': ufos['latitude'],
        'Longitude': ufos['longitude']
    })
    ufos.dropna(inplace=True)
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

    X = ufos[['Seconds', 'Latitude', 'Longitude']]
    y = ufos['Country']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    return f"Model trained with accuracy: {accuracy}"

if __name__ == "__main__":
    app.run(debug=True)
