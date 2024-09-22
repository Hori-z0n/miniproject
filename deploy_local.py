from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

file = 'heart_failure2.pkl'
model = pickle.load(open(file, 'rb'))

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:

        Sex = int(request.form["sex"])
        ChestPainType = int(request.form["chestpaintyp"])
        RestingECG = int(request.form["restingecg"])
        ExerciseAngina = int(request.form["exerciseangina"])
        ST_Slope = int(request.form["st_slope"])
        Age = float(request.form["age"])
        RestingBP = float(request.form["restingbp"])
        Cholesterol = float(request.form["cholesterol"])
        FastingBS = float(request.form["fastingbs"])
        MaxHR = float(request.form["maxhr"])
        Oldpeak = float(request.form["oldpeak"])

        
        X = np.array([[Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak]])

        
        prediction = model.predict(X)

        
        result = 'มีความเสี่ยงหัวใจล้มเหลว' if prediction[0] == 1 else 'ไม่มีความเสี่ยงหัวใจล้มเหลว'

        return f"Prediction: {result}"
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
