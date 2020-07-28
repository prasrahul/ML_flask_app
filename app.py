from flask import Flask,request,render_template
from flask_cors import  cross_origin
import pickle
import os

app =Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template("home.html")


@app.route('/predict',methods=['GET', 'POST'])
@cross_origin()
def result():
    #  reading the inputs given by the user
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])


    filename = 'modelForPrediction.sav'
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    scalar = pickle.load(open("sandardScalar.sav", 'rb'))
    # predictions using the loaded model file
    prediction = loaded_model.predict(scalar.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age]]))
    print('prediction is', prediction)
    if prediction ==[1]:
            prediction = "diabetes"

    else:
            prediction = "Normal"

    # showing the prediction results in a UI
    if  prediction =="diabetes":

        return render_template('diabetes.html', prediction=prediction)
    else:
        return render_template('Normal.html',prediction=prediction)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    #app.run(debug=True)
