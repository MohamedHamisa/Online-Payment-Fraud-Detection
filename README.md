

# Online Payment Fraud Detection

This project aims to detect fraudulent online payment transactions using machine learning techniques. The dataset used contains information about transactions, including features that describe each transaction and whether it is fraudulent or not.

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#eda)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)

## Dataset

The dataset (`online_payment_fraud_detection.csv`) consists of the following columns:

- `step`: Unit of time in the real world (1 step = 1 hour).
- `type`: Type of transaction (e.g., CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).
- `amount`: Transaction amount.
- `nameOrig`: Origin account.
- `oldbalanceOrg`, `newbalanceOrig`: Original and new balance before and after the transaction for origin account.
- `nameDest`: Destination account.
- `oldbalanceDest`, `newbalanceDest`: Original and new balance before and after the transaction for destination account.
- `isFraud`: Binary indicator of whether the transaction is fraudulent (1) or not (0).

## Features

- Exploratory Data Analysis (EDA) provides insights into the distribution of fraudulent vs. non-fraudulent transactions, transaction amounts, and more.

## Exploratory Data Analysis (EDA)

Exploratory data analysis examines the distribution and characteristics of the dataset. Visualizations such as count plots and box plots are used to understand the data.

## Data Preprocessing

Data preprocessing includes handling missing values, feature engineering (creating new features like balance differences), and encoding categorical variables (one-hot encoding 'type').

## Model Training

The RandomForestClassifier model with 100 estimators is trained on the preprocessed data to predict fraudulent transactions based on transaction details.

## Model Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Visualizations like confusion matrix and ROC curve are used to assess the model's effectiveness in detecting fraud.

## Conclusion

This project demonstrates the process of building a machine learning model for online payment fraud detection. It highlights the importance of data preprocessing, model selection, and evaluation in tackling real-world challenges like fraud detection.
