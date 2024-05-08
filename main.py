import pandas as pd
import joblib
from preprocessing import clean_data, get_clientnum, get_existing_customers, get_attrited_customers
from predict import predict_probability, split_X_y, prediction_as_json

clean_data()

model = joblib.load('model.joblib')

df = pd.read_csv('assets/cleaned_data.csv')

clientnums_df = get_clientnum()

df_not_attrited_customers = df[df['Attrition_Flag'] == 0]

df_attrited_customers = df[df['Attrition_Flag'] == 1]

print(f'Number of existing customers: {get_existing_customers().shape[0]}')
print(f'Number of attrited customers: {get_attrited_customers().shape[0]}')

X, y = split_X_y(get_existing_customers())

predict_probability(X)

prediction_as_json(X)
