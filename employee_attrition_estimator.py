# -*- coding: utf-8 -*-
"""Employee_Attrition_Estimator

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13wJbOPzpcsaXeJLuL_qLhpYu0lF6XK62
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
import joblib

# Importing Test Data & Dropping some Columns ----------------------------------------------------------------------- #
missing = [' ', 'none', 'NaN', 'nan', 'null', 'empty', 'missing']
df = pd.read_csv('Employee_Attrition.csv',na_values = missing)
df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
df.drop(columns=['MonthlyRate', 'HourlyRate', 'DailyRate'],inplace = True)
df.drop(columns=['EmployeeNumber'],inplace = True)
# ------------------------------------------------------------------------------------------------------------------ #

# Winsorization for Outlier Handling ------------------------------------------------------------------------------- #
from scipy.stats.mstats import winsorize
lower_percentile = 0.05
upper_percentile = 0.05
for column in df.columns:
    if df[column].dtype == 'int64':
        winsorized_values = winsorize(df[column], limits=(lower_percentile, upper_percentile), inclusive=(True, True))
        df[column] = winsorized_values.data
# ------------------------------------------------------------------------------------------------------------------ #

# Standard Scaling and Label Encoding for Standardizing and Binary Encoding ---------------------------------------- #
from sklearn.preprocessing import StandardScaler, LabelEncoder
ss = StandardScaler()
le = LabelEncoder()
scalers = {}
for column in df.select_dtypes(include=['int']).columns:
    scalers[column] = StandardScaler()
    df[column] = scalers[column].fit_transform(df[column].values.reshape(-1, 1))
df['Attrition'] = le.fit_transform(df['Attrition'])
df['BusinessTravel'] = le.fit_transform(df['BusinessTravel'])
df['Department'] = le.fit_transform(df['Department'])
df['EducationField'] = le.fit_transform(df['EducationField'])
df['Gender'] = le.fit_transform(df['Gender'])
df['JobRole'] = le.fit_transform(df['JobRole'])
df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
df['OverTime'] = le.fit_transform(df['OverTime'])
# ------------------------------------------------------------------------------------------------------------------ #

# Principal Component Analysis for High Dimensionality ------------------------------------------------------------- #
from sklearn.decomposition import PCA
column_names = df.columns.tolist()
pca = PCA()
pca.fit(df)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance_ratio >= 0.99) + 1  # Keep 99% of variance
pca = PCA(n_components)
principal_components = pca.fit_transform(df)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
# ------------------------------------------------------------------------------------------------------------------ #

ID_array =[None] * len(df)
for index, row in enumerate(df.iterrows(), start=0):
  ID_array[index] = "EM" + str(index)
df['Employee Code'] = ID_array

# Specifying names for dataframes used ----------------------------------------------------------------------------- #
df_for_X = principal_df
df_for_Y = df
# ------------------------------------------------------------------------------------------------------------------ #

# Specifying best amount of K value -------------------------------------------------------------------------------- #
best = 10
# ------------------------------------------------------------------------------------------------------------------ #

# Hyperparameter Tuned SVM, with PCA

X = df_for_X.copy()
y = df_for_Y['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('undersample', RandomUnderSampler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=best)),
    ('smote', SMOTE(sampling_strategy='auto', k_neighbors=20, random_state=42)),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'svm__gamma': [0.1, 1, 10]
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict(X_test)

result = grid_search.best_estimator_.predict(X)
estimation = pd.DataFrame(result)
estimation['Employee ID'] = df['Employee Code']
estimation.rename(columns={"0" : "Prediction"}, inplace=True)
estimation.columns.values[0] = 'Estimation'
estimation = estimation[['Employee ID', 'Estimation']]

num_rows_yes = estimation['Estimation'].value_counts()[1]
num_rows_nop = estimation['Estimation'].value_counts()[0]
amonts = [int(num_rows_yes),int(num_rows_nop)]
labels = ['Yes','No']
plt.pie(amonts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.show()

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'model_SVM.pkl')

# Hyperparameter Tuned Logistic Regression, with PCA

X = df_for_X.copy()
y = df_for_Y['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('undersample', RandomUnderSampler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=best)),
    ('smote', SMOTE(sampling_strategy='auto', k_neighbors=20, random_state=42)),
    ('logistic', LogisticRegression(max_iter=1000))
])

param_grid = {
    'logistic__C': [0.1, 1, 10, 100],
    'logistic__penalty': ['l1', 'l2'],
    'logistic__solver': ['liblinear']
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict(X_test)

result = grid_search.best_estimator_.predict(X)
estimation = pd.DataFrame(result)
estimation['Employee ID'] = df['Employee Code']
estimation.rename(columns={"0" : "Prediction"}, inplace=True)
estimation.columns.values[0] = 'Estimation'
estimation = estimation[['Employee ID', 'Estimation']]

num_rows_yes = estimation['Estimation'].value_counts()[1]
num_rows_nop = estimation['Estimation'].value_counts()[0]
amonts = [int(num_rows_yes),int(num_rows_nop)]
labels = ['Yes','No']
plt.pie(amonts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.show()

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'model_Log.pkl')

# Hyperparameter Tuned Perceptron, with PCA

X = df_for_X.copy()
y = df_for_Y['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('undersample', RandomUnderSampler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=best)),
    ('smote', SMOTE(sampling_strategy='auto', k_neighbors=20, random_state=42)),
    ('perceptron', Perceptron())
])

param_grid = {
    'perceptron__alpha': [0.0001, 0.001, 0.01, 0.1],
    'perceptron__max_iter': [100, 500, 1000],
    'perceptron__penalty': ['l2', 'l1', 'elasticnet'],
    'perceptron__tol': [1e-3, 1e-4, 1e-5]
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict(X_test)

result = grid_search.best_estimator_.predict(X)
estimation = pd.DataFrame(result)
estimation['Employee ID'] = df['Employee Code']
estimation.rename(columns={"0" : "Prediction"}, inplace=True)
estimation.columns.values[0] = 'Estimation'
estimation = estimation[['Employee ID', 'Estimation']]

num_rows_yes = estimation['Estimation'].value_counts()[1]
num_rows_nop = estimation['Estimation'].value_counts()[0]
amonts = [int(num_rows_yes),int(num_rows_nop)]
labels = ['Yes','No']
plt.pie(amonts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.show()

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'model_Per.pkl')

print(sns.__version__)
print(imblearn.__version__)