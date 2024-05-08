import pandas as pd
import joblib
from preprocessing import clean_data, get_clientnum, get_existing_customers, get_attrited_customers
from predict import split_X_y, predict_and_save_as_json

# clean the data and loads clean data as a pandas dataframe
clean_data()
df = pd.read_csv('assets/cleaned_data.csv')

# loads the model trained in model.py file
model = joblib.load('model.joblib')

print(f'Number of existing customers: {get_existing_customers().shape[0]}')
print(f'Number of attrited customers: {get_attrited_customers().shape[0]}')

# split the data of existing customers
X_existing, y = split_X_y(get_existing_customers())

# predicts the probability of being attrited and saves it as a json file
predict_and_save_as_json(
    X_existing, threshold=0.5,
    file_name='At_Risk_Customers.json',
    customer_type='existing'
    )


X_attrited, y = split_X_y(get_attrited_customers())

predict_and_save_as_json(
    X_attrited, threshold=0.5,
    file_name='Reactivation_Customer.json',
    customer_type='attrited'
    )
