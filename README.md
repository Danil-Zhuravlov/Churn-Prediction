# Churn Prediction with Credit Card Customers Dataset

Predicting customer churn is essential for retaining valuable clients. This project leverages machine learning techniques to identify at-risk customers and proposes strategies for retention.

## Table of Contents
- [Introduction](#introduction-📝) 📝
- [Project Goals](#project-goals-🎯) 🎯
- [Setup Instructions](#setup-instructions-⚙️) ⚙️
- [Project Structure](#project-structure-📂) 📂
- [Usage](#usage-💻) 💻
- [Results](#results-📊) 📊
- [Future Improvements](#future-improvements-🔮) 🔮
- [About](#about-ℹ️) ℹ️

## Introduction 📝
This project focuses on predicting customer churn in a bank using data from the Credit Card Customers Dataset available on Kaggle. The dataset provides information about various features of customers, allowing us to build a predictive model to identify those likely to close their accounts.

## Project Goals 🎯
1. **Marketing Focus**: Identify clients more likely to close their bank accounts to prevent attrition.
2. **Model Development**: Develop machine learning models for classification to predict customer churn.
3. **Risk Assessment**: Predict the similarity of each customer to different categories to prioritize retention efforts.

## Setup Instructions ⚙️
To run this project locally, follow these steps:
1. Clone this repository.
2. Install the required libraries listed in `requirements.txt`.
```
pip install -r requirements.txt
```
3. Run `main.py` to obtain a JSON file with a list of at-risk customers and reactivation candidates.
4. For model retraining, execute `model.py`.
5. It is recommended to use a Python virtual environment to avoid conflicts.

## Project Structure 📂
```
churn-prediction/
│
├── data/
│ └── credit_card_customers.csv
├── models/
│ └── RandomForestClassifier.joblib
├── README.md
├── main.py
├── model.py
└── requirements.txt
```

## Usage 💻
Once set up, the project can be utilized as follows:
- **`main.py`**: Generates JSON files with customer similarity scores.
- **`model.py`**: Retrains the RandomForestClassifier model.

## Results 📊
- **Model Performance**:
  - Accuracy: 0.9785
  - Confusion Matrix:
    ```
    [[1628   48]
     [  25 1699]]
    ```
- **Client Similarity Prediction**: Provides a JSON file with similarity scores for each client.

## Future Improvements 🔮
- Create a better data preprocessing pipeline to make it easier to configure and maintain.
- Provide a better file structure for the project.
- Enhance model performance by exploring different algorithms and feature engineering techniques.

## About ℹ️
This project was completed in 5 days as part of the Data Science course at BeCode. It addresses the challenge of customer churn prediction using machine learning techniques.
