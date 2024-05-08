import pandas as pd
import numpy as np
import joblib
import json
from preprocessing import get_clientnum

model = joblib.load('model.joblib')

def split_X_y(dataframe):
    X = dataframe.drop(['Attrition_Flag'], axis=1)
    y = dataframe['Attrition_Flag']
    return X, y

def predict_probability(X):
    y_pred_proba = model.predict_proba(X)
    probabilities = y_pred_proba[:, 1]
    high_probabilities = probabilities > 0.5
    num_high_samples = np.sum(high_probabilities)
    print("Number of samples with churn probability > 50%:", num_high_samples)

def prediction_as_json(X):
    clientnums = get_clientnum()['CLIENTNUM']
    probas = model.predict_proba(X)
    result_dict = {clientnum: proba[1] for clientnum, proba in zip(clientnums, probas)}

    result = dict((k, v) for k, v in result_dict.items() if v > 0.6)

    # saves the result to the .json file
    with open('result.json', 'w') as f:
        json.dump(result, f)
