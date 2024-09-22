from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

history = model.fit(X, y)

score = model.score(X_test, y_test)

# print(X_test)
# print(y_test)
# Sex  ChestPainType  RestingECG  ExerciseAngina  ST_Slope  Age  RestingBP  Cholesterol  FastingBS  MaxHR  Oldpeak
X_test_1 = [[1,0,1,1,2,60,130,253,0,144.0,1.4]] # 1
X_test_2 = [[1,1,1,0,2,35,122,192,0,174,0.0]] # 0
# print(type(np.array(X_test)[0][0]))
@app.route('/')
def home():
    return render_template('index.html')

# GET: A GET message is send, and the server returns data
# POST: Used to send HTML form data to the server
# ADD Post method to the decorator to allow for form submission.
# Redirect to /predict page with the output
@app.route('/predict', methods=["POST"])
def predict():
    print("Test score: {0:.2f} %".format(100 * score))
    Sex = request.form["sex"]
    ChestPainType = request.form["chestpaintyp"]
    RestingECG = request.form["restingecg"]
    ExerciseAngina = request.form["exerciseangina"]
    ST_Slope = request.form["st_slope"]
    Age = request.form["age"]
    RestingBP = request.form.get("restingbp")
    Cholesterol = request.form.get("cholesterol")
    FastingBS = request.form.get("fastingbs")
    MaxHR = request.form.get("maxhr")
    if(int(MaxHR)<60):
        print("Low Heart rate")
    elif(int(MaxHR)>202):
        print("High Heart rate")

    Oldpeak = request.form.get("oldpeak")
    # output = model.predict(feature)
    X = [[int(Sex), int(ChestPainType), int(RestingECG),int(ExerciseAngina), int(ST_Slope), int(Age), int(RestingBP), int(Cholesterol), int(FastingBS), int(MaxHR), float(Oldpeak)]]
    # X = f"{Sex} {ChestPainType} {RestingECG} {ExerciseAngina} {ST_Slope} {Age} {RestingBP} {Cholesterol} {FastingBS} {MaxHR} {Oldpeak}"
    # index = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    # df = pd.DataFrame(X, columns=index)

    prediction = model.predict(X)
    y_pred = np.where(prediction>0.5, 1, 0)
    print(type(Sex))
    print(type(Age))
    print(prediction)
    print(y_pred)
    result = ""
    if int(prediction) == 1:
        result = 'มีความเสี่ยงหัวใจล้มเหลว'
    else:
        result = 'ไม่มีความเสี่ยงหัวใจล้มเหลว'

    return render_template('index.html', prediction_text = "you value = {} predict {} : {} : {}".format(X, prediction, y_pred, result))

if __name__=='__ main__':
    app.run(debug=True)
