#from flask import Flask, request, redirect, url_for, jsonify,render_template
from flask import Flask, request, url_for, jsonify,render_template
from flask_cors import CORS, cross_origin
import numpy as np
# import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

global knn
global modelfile

#Saving & Loading sklearn models
import pickle

app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

modelfile = "models/knn_model.pickle"

#inspect the head 
head = datasets.load_iris().data

#Load the iris dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)
np.unique(iris_y)

#features
features = datasets.load_iris().feature_names
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

#target_names
target_names = datasets.load_iris().target_names
#['setosa' 'versicolor' 'virginica']


# #Split the datasets into train/test
# # X & y train and X& y test
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X,iris_y,test_size=0.5,random_state=42,shuffle=True,stratify=iris_y)

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/head')
@cross_origin()
def data():
    return jsonify({0:np.array2string(head)})

@app.route('/v1/train',methods=['GET', 'POST'])
@cross_origin()
def train():
    #### Training the model
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train,iris_y_train)
    pickle.dump(knn,open(modelfile,'wb'))
    result = {0:"Training was successful"}
    return jsonify(result)

@app.route('/v1/predict',methods=['GET', 'POST'])
@cross_origin()
def predict():
    # Predict
    knn = pickle.load(open(modelfile, 'rb')) 
    predictions = knn.predict(iris_X_test[:5])
    results = []
    for index,x in enumerate(predictions):
        if predictions[index] == 0:
            results.append(target_names[0])
        elif predictions[index] == 1:
            results.append(target_names[1])
        elif predictions[index] == 2:
            results.append(target_names[2])
    return jsonify(results)

if __name__ == "__main__":
    app.run(threaded=True,host='0.0.0.0')