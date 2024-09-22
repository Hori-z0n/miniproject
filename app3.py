from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
app = Flask(__name__)

data = pd.read_csv('.././heart_disease2.csv')
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Normalization
# sc = StandardScaler()
# X_sc = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12)

# Build Model
model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')

# GET: A GET message is send, and the server returns data
# POST: Used to send HTML form data to the server
# ADD Post method to the decorator to allow for form submission.
# Redirect to /predict page with the output
@app.route('/predict', methods=["POST"])
def predict():
    
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
    # X = [[Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak]]
    X = f"{Sex} {ChestPainType} {RestingECG} {ExerciseAngina} {ST_Slope} {Age} {RestingBP} {Cholesterol} {FastingBS} {MaxHR} {Oldpeak}"
    # index = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    # df = pd.DataFrame(X, columns=index)

    # prediction = model.predict(X)

    # result = ""
    # if int(prediction) == 1:
    #     result = 'มีความเสี่ยงหัวใจล้มเหลว'
    # else:
    #     result = 'ไม่มีความเสี่ยงหัวใจล้มเหลว'

    return render_template('index.html', prediction_text = "{}".format(X))

if __name__=='__ main__':
    app.run(debug=True)
