# heart_disease_classification
This is heart disease classification API for self-learning purpose


This is simple implementation of ML classification and API development for self learning and future development. 

The main data came from the UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/heart+Disease

There is also on Kaggle. 
https://www.kaggle.com/ronitf/heart-disease-uci


How to use this API.

You can send request to first API "/predict" route, the request must be in this format.
For example

{
    "age": 63.0,
    "sex": 1.0,
    "cp": 3.0,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1.0,
    "restecg": 0.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 2.3,
    "slope": 0.0,
    "ca": 0.0,
    "thal": 1.0
    }


And as response you will get classification result, where [1] mean that person has heart disease and [0] mean that person hasn't heart disease.

The second part of API is "/train" route, which train new model base on your data and save trained model by given name.
For example

{"model_name": "your trained model name"}


(Just I want to mention that this is just a for self learning and I am planning add new features)

