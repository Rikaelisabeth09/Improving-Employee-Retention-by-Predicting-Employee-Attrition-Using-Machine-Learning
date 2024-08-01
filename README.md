# Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning

## Table of Contents

1. [Project Overview](#project-overview)
2. [Author](#author)
3. [Motivation](#motivation)
4. [Data Preprocessing](#data-preprocessing)
   - [Handling Missing Values](#handling-missing-values)
   - [Removing Duplicates](#removing-duplicates)
   - [Categorical Feature Encoding](#categorical-feature-encoding)
   - [Feature Engineering](#feature-engineering)
5. [Machine Learning Models](#machine-learning-models)
6. [Key Insights](#key-insights)
7. [Recommendations](#recommendations)


## Project Overview

This project aims to predict employee attrition using machine learning techniques. By analyzing various factors that influence employee satisfaction and retention, the project provides actionable insights to help organizations reduce turnover and retain top talent.


## Author

- **Rika Elisabeth**
  - [LinkedIn](https://www.linkedin.com/in/rikaelisabeth/)

## Motivation

Human resources are a critical asset for any company. Understanding and predicting employee attrition can help in designing better HR strategies, ultimately saving costs associated with recruitment and training.

## Data Preprocessing

### Handling Missing Values
"```python
import pandas as pd
df = pd.read_csv('employee_data.csv')
df.fillna(df.mean(), inplace=True) ```

### Removing Duplicates
"```python
df.drop_duplicates(inplace=True)```

### Categorical Feature Encoding
"```python
df = pd.get_dummies(df, columns=['CategoryColumn'])```

### Feature Engineering
"```python
ddf['NewFeature'] = df['ExistingFeature'] * 2```

### Machine Learning Models
"```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'ROC AUC: {roc_auc}')```

## Key Insights
1. High attrition rates were observed in specific divisions, such as Data Analysts and Front-End Engineers, indicating potential issues in these areas. For instance, 100% of employees resigning from the Data Analyst division were fresh graduates, with major reasons being toxic culture and internal conflict.
2. Key reasons for employee resignations included toxic culture, internal conflict, and lack of career progression. Toxic culture alone accounted for 75% of the resignations among fresh graduates in the Data Analyst division.
3. Despite a high resignation rate, 50% of employees in the Data Analyst division had excellent performance, suggesting that the organization might be losing valuable talent due to unresolved issues.
4. There was a significant decrease in the number of employees from 2015 to 2020, highlighting potential challenges in retaining talent during this period. Further investigation is needed to understand the underlying causes.

## Recommendations
1. Onboarding and Training: Improve the onboarding experience for new hires, especially fresh graduates.
2. Work Environment: Address toxic culture and improve the overall work environment.
3. Career Development: Provide clear career progression paths and skill development opportunities.
4. Recognition and Rewards: Implement a robust recognition program for high performers.
