from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)   
model_path = os.path.join(BASE_DIR, "iris_model.pkl")

model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            sl = float(request.form.get("sepal_length"))
            sw = float(request.form.get("sepal_width"))
            pl = float(request.form.get("petal_length"))
            pw = float(request.form.get("petal_width"))

            features = np.array([[sl, sw, pl, pw]])
            pred = model.predict(features)[0]

            classes = ["Setosa ", "Versicolor", "Virginica"]
            prediction = classes[pred]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)