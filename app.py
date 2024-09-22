from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn import metrics, linear_model
import pandas as pd
app = Flask(__name__)

filename = 'heart_failure.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)
# model = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# GET: A GET message is send, and the server returns data
# POST: Used to send HTML form data to the server
# ADD Post method to the decorator to allow for form submission.
# Redirect to /predict page with the output
@app.route('/predict', methods=["POST"])
def predict():
    # Sex = request.form["sex"]
    # ChestPainType = request.form["chestpaintyp"]
    # RestingECG = request.form["restingecg"]
    # ExerciseAngina = request.form["exerciseangina"]
    # ST_Slope = request.form["st_slope"]
    # Age = request.form.get("age")
    # RestingBP = request.form.get("restingbp")
    # Cholesterol = request.form.get("cholesterol")
    # FastingBS = request.form.get("fastingbs")
    # MaxHR = request.form.get("maxhr")
    # Oldpeak = request.form.get("oldpeak")

    # Sex = int(request.form["sex"])
    # ChestPainType = int(request.form["chestpaintyp"])
    # RestingECG = int(request.form["restingecg"])
    # ExerciseAngina = int(request.form["exerciseangina"])
    # ST_Slope = int(request.form["st_slope"])
    # Age = float(request.form["age"])
    # RestingBP = float(request.form["restingbp"])
    # Cholesterol = float(request.form["cholesterol"])
    # FastingBS = float(request.form["fastingbs"])
    # MaxHR = float(request.form["maxhr"])
    # Oldpeak = float(request.form["oldpeak"])
    # init_feature = [float(x) for x in request.form.values()]
    # feature = [np.array(init_feature)]
    Sex = int(request.form["sex"])
    ChestPainType = int(request.form["chestpaintyp"])
    RestingECG = int(request.form["restingecg"])
    ExerciseAngina = int(request.form["exerciseangina"])
    ST_Slope = int(request.form["st_slope"])
    Age = float(request.form.get("age"))
    RestingBP = float(request.form.get("restingbp"))
    Cholesterol = float(request.form.get("cholesterol"))
    FastingBS = float(request.form.get("fastingbs"))
    MaxHR = float(request.form.get("maxhr"))
    Oldpeak = float(request.form.get("oldpeak"))
    # output = model.predict(feature)
    X = np.array([[Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak]])
    # index = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    # df = pd.DataFrame(X, columns=index)

    prediction = model.predict(X)

    # result = ""
    # if int(prediction) == 1:
    #     result = 'มีความเสี่ยงหัวใจล้มเหลว'
    # else:
    #     result = 'ไม่มีความเสี่ยงหัวใจล้มเหลว'

    return render_template('index.html', prediction_text = "{}".format(prediction))

if __name__=='__ main__':
    app.run(debug=True)
