import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc



# Load dataset
df = pd.read_csv('online_payment_fraud_detection.csv')

# Display first few rows
print(df.head())


### Step 3: Exploratory Data Analysis (EDA)


# Basic info about the dataset
print(df.info())
print(df.describe())
print(df.isnull().sum())


#### Distribution of Fraud vs. Non-Fraud Transactions


plt.figure(figsize=(8, 6))
sns.countplot(x='isFraud', data=df)
plt.title('Distribution of Fraud vs. Non-Fraud Transactions')
plt.show()


#### Amount Distribution for Fraud and Non-Fraud Transactions


plt.figure(figsize=(12, 6))
sns.boxplot(x='isFraud', y='amount', data=df)
plt.yscale('log')
plt.title('Transaction Amount Distribution for Fraud and Non-Fraud Transactions')
plt.show()


### Step 4: Data Preprocessing

#### Handling Missing Values


# Check for missing values
print(df.isnull().sum())
# No missing values found in this dataset based on info and describe outputs.


#### Feature Engineering

# Dropping columns that are not useful for the model
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Adding new features (e.g., balance difference before and after the transaction)
df['balanceOrigDiff'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balanceDestDiff'] = df['newbalanceDest'] - df['oldbalanceDest']

# Dropping original balance columns
df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], axis=1, inplace=True)


#### Encoding Categorical Variables

# One-hot encoding for the 'type' column
df = pd.get_dummies(df, columns=['type'], drop_first=True)


### Step 5: Train-Test Split

X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


### Step 6: Feature Scaling


# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


### Step 7: Model Training

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


### Step 8: Model Evaluation

#### Predictions and Classification Report

# Predictions
y_pred = model.predict(X_test_scaled)

# Classification report
print(classification_report(y_test, y_pred))


#### Confusion Matrix


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#### ROC Curve and AUC

# ROC Curve and AUC
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()




