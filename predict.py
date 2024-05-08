import pandas as pd
import numpy as np
import joblib
import json
from preprocessing import get_clientnum

model = joblib.load('model.joblib')

def predict_and_save_as_json(X, threshold=0.5, file_name='result.json', customer_type='existing'):
    """
    Predicts the probability of churn for each customer in the input data and saves the results as a JSON file.

    Args:
        X (array-like): Input data containing features for each customer.
        threshold (float, optional): Threshold value for filtering the predicted probabilities. Defaults to 0.5.
        file_name (str, optional): Name of the JSON file to save the results. Defaults to 'result.json'.
        customer_type (str, optional): Type of customer ('existing' or 'attrited'). Defaults to 'existing'.

    Returns:
        None
    """
    if customer_type == 'existing':
        Attrited_Flag_value = 1
    
    elif customer_type == 'attrited':
        Attrited_Flag_value = 0

    else:
        print('Customer type can be "existing" or "attrited" only. Try again.')
    
    clientnums = get_clientnum()['CLIENTNUM']
    probas = model.predict_proba(X)
    result_dict = {clientnum: proba[Attrited_Flag_value] for clientnum, proba in zip(clientnums, probas)}

    result = dict((k, v) for k, v in result_dict.items() if v > threshold)

    # saves the result to the .json file
    with open(file_name, 'w') as f:
        json.dump(result, f)
