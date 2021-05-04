import json
import logging
import os
import joblib
import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "incorrect_range": 
    {"fixed acidity": 7897897, 
    "volatile acidity": 555, 
    "citric acid": 99, 
    "residual sugar": 99, 
    "chlorides": 12, 
    "free sulfur dioxide": 789, 
    "total sulfur dioxide": 75, 
    "density": 2, 
    "pH": 33, 
    "sulphates": 9, 
    "alcohol": 9
    },

    "correct_range":
    {"fixed acidity": 5.4, 
    "volatile acidity": 0.53, 
    "citric acid": 0.16, 
    "residual sugar": 2.7, 
    "chlorides": 0.036, 
    "free sulfur dioxide": 34, 
    "total sulfur dioxide": 128, 
    "density": 0.98856, 
    "pH": 3.2, 
    "sulphates": 0.53, 
    "alcohol": 13.2
    },

    "incorrect_col":
    {"fixed_acidity": 5, 
    "volatile_acidity": 1, 
    "citric_acid": 0.5, 
    "residual_sugar": 10, 
    "chlorides": 0.5, 
    "free sulfur dioxide": 3, 
    "total_sulfur dioxide": 75, 
    "density": 1, 
    "pH": 3, 
    "sulphates": 1, 
    "alcohol": 9
    }
}

TARGET_range = {
    "min": 3.0,
    "max": 9.0
}

def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert  TARGET_range["min"] <= res <= TARGET_range["max"]

def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert  TARGET_range["min"] <= res["response"] <= TARGET_range["max"]

def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)

def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message