from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scalr = StandardScaler()
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    salary = int(request.form['salary'])

    input_data = pd.Dataframe({'Gender':[gender],'Age':[age],'Salary':[salary]})

    scaled_data = scalr.fit_transform(input_data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        output = "purchased"


    else:
        output = "not purchased"

    return render_template('index.html',message = output)    

if __name__ == "__main__":
    app.run(debug=True)       

