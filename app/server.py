from flask import Flask,render_template,request, redirect, url_for,flash,jsonify, Response, json
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import io
import os
import requests
import pickle
from sklearn.externals import joblib
from flask_cors import CORS

#Let's define some variables
lr = joblib.load("model/regularized_model.pkl") # Load the elasticNet model
print ('Model loaded')
model_columns = joblib.load("model/model_columns.pkl")


app = Flask(__name__)
CORS(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
#Checking if the extension is a valid one
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')
#Route pour le modèle de prédiction
@app.route('/predict',methods=['GET','POST'])
def predict_price():
    tv = request.form["tv"]
    radio = request.form["radio"]
    journal = request.form["journal"]
    json_ = {
   "TV":float(tv),
   "radio":float(radio),
   "newspaper":float(journal)
}
    print(json_)
    #return jsonify((json_))
    if lr:
        try:
            print("IN")
            print(json_)
            query = pd.DataFrame([json_])
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            prediction = list(lr.predict(query))
            result = float(prediction[0])
            return render_template("index.html",result = result)

        except:

            return "Not available data"
    else:
        print ('Train the model first')
        return ('No model here to use')

#The main function
if __name__=="__main__":

    print("the server is up...")
    app.run(host='0.0.0.0',debug=True)
    