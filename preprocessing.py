import numpy as np
import pandas as pd
import os



def get_clientnum(file_path='assets/BankChurners.csv'):
    df = pd.read_csv(file_path)
    clientnums_df = df[['CLIENTNUM']]
    return clientnums_df

def get_existing_customers(file_path='assets/cleaned_data.csv'):
    df = pd.read_csv(file_path)
    existing_customers = df[df['Attrition_Flag'] == 0]
    return existing_customers

def get_attrited_customers(file_path='assets/cleaned_data.csv'):
    df = pd.read_csv(file_path)
    attrited_customers = df[df['Attrition_Flag'] == 1]
    return attrited_customers



def split_X_y(file_path='assets/cleaned_data.csv'):
    dataframe = pd.read_csv(file_path)
    X = dataframe.drop(['Attrition_Flag'], axis=1)
    y = dataframe['Attrition_Flag']
    return X, y



def clean_data(file_path='assets/BankChurners.csv'):

    pd.set_option('future.no_silent_downcasting', True)

    df = pd.read_csv(file_path)

    df.drop([
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'CLIENTNUM'],
        axis=1,
        inplace=True)

    education_level = {
        'Uneducated':0, 'High School':1, 'College':2, 'Graduate':3,
        'Post-Graduate':4, 'Doctorate':5, 'Unknown':6
        }
    df['Education_Level'] = df['Education_Level'].replace(education_level)

    marital_status = {'Single':0, 'Married':1, 'Divorced':2, 'Unknown':3}
    df['Marital_Status'] = df['Marital_Status'].replace(marital_status)

    income_category = {
        'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2,
        '$80K - $120K': 3, '$120K +': 4, 'Unknown': 5
        }
    df['Income_Category'] = df['Income_Category'].replace(income_category)

    attrition_flag = {'Existing Customer': 0, 'Attrited Customer': 1}
    df['Attrition_Flag'] = df['Attrition_Flag'].replace(attrition_flag)

    gender = {'F': 0, 'M': 1}
    df['Gender'] = df['Gender'].replace(gender)

    card_category = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    df['Card_Category'] = df['Card_Category'].replace(card_category)

    # Get the current working directory
    current_dir = os.getcwd()

    # Get the path to the assets folder
    assets_dir = os.path.join(current_dir, 'assets')

    # Create the assets folder if it doesn't exist
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # Save the DataFrame to a CSV file inside the assets folder
    df.to_csv(os.path.join(assets_dir, 'cleaned_data.csv'), index=False)
