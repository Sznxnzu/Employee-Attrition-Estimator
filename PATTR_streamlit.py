import streamlit as st
import joblib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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



# Specifying names for dataframes used ----------------------------------------------------------------------------- #
df_for_X = principal_df
df_for_Y = df
# ------------------------------------------------------------------------------------------------------------------ #

ID_array =[None] * len(df)
for index, row in enumerate(df.iterrows(), start=0):
  ID_array[index] = "EM" + str(index)
df['Employee Code'] = ID_array

# Specifying best amount of K value -------------------------------------------------------------------------------- #
best = 10
# ------------------------------------------------------------------------------------------------------------------ #



# Loading Models --------------------------------------------------------------------------------------------------- #
model_SVM = joblib.load('model_SVM.pkl')
model_Log = joblib.load('model_Log.pkl')
model_Per = joblib.load('model_Per.pkl')
# ------------------------------------------------------------------------------------------------------------------ #

def create_pie_chart(data, labels):
    fig, ax = plt.subplots()
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal') 
    return fig

def make_prediction_and_plot(model_selection):
    
    input_data = df
    
    if model_selection == 'SVM':
        model = model_SVM
        
        X = df_for_X.copy()
        y = df_for_Y['Attrition']

        result = model.predict(X)
        estimation = pd.DataFrame(result)
        estimation['Employee ID'] = df['Employee Code']
        estimation.rename(columns={"0" : "Prediction"}, inplace=True)
        estimation.columns.values[0] = 'Estimation'
        estimation = estimation[['Employee ID', 'Estimation']]

        num_rows_yes = estimation['Estimation'].value_counts()[1]
        num_rows_nop = estimation['Estimation'].value_counts()[0]
        amonts = [int(num_rows_yes),int(num_rows_nop)]
        labels = ['Yes','No']
        
        fig = create_pie_chart(amonts,labels)
        
        return fig
        
        
    elif model_selection == 'Logistic':
        model = model_Log
        
        X = df_for_X.copy()
        y = df_for_Y['Attrition']
        
        result = model.predict(X)
        estimation = pd.DataFrame(result)
        estimation['Employee ID'] = df['Employee Code']
        estimation.rename(columns={"0" : "Prediction"}, inplace=True)
        estimation.columns.values[0] = 'Estimation'
        estimation = estimation[['Employee ID', 'Estimation']]

        num_rows_yes = estimation['Estimation'].value_counts()[1]
        num_rows_nop = estimation['Estimation'].value_counts()[0]
        amonts = [int(num_rows_yes),int(num_rows_nop)]
        labels = ['Yes','No']
        
        fig = create_pie_chart(amonts,labels)
        
        return fig
        
    elif model_selection == 'Perceptron':
        model = model_Per
        
        X = df_for_X.copy()
        y = df_for_Y['Attrition']
        
        result = model.predict(X)
        estimation = pd.DataFrame(result)
        estimation['Employee ID'] = df['Employee Code']
        estimation.rename(columns={"0" : "Prediction"}, inplace=True)
        estimation.columns.values[0] = 'Estimation'
        estimation = estimation[['Employee ID', 'Estimation']]

        num_rows_yes = estimation['Estimation'].value_counts()[1]
        num_rows_nop = estimation['Estimation'].value_counts()[0]
        amonts = [int(num_rows_yes),int(num_rows_nop)]
        labels = ['Yes','No']
        
        fig = create_pie_chart(amonts,labels)
        
        return fig



def main():
    
    def make_prediction():
        st.session_state.button_clicked = True
        
    st.title("Employee Attrition Estimator")
    
    st.subheader('Employee Attrition')
    st.text('Employee Attrition is defined as employees leaving their organizations for unpredictable or uncontrollable reasons. Many terms make up attrition, the most common being termination, resignation, planned or voluntary retirement, structural changes, long-term illness, layoffs.')
    st.text('This, is certainly a problem for many companies and organizations, especially with the rise of employee attrition worldwide.')
    st.text('As such, an estimator has been made based on the dataset from IBM.')

    st.subheader('How does it work?')
    st.text('This estimator uses three methods with varying accuracies, which are Support Vector Machines, Perceptrons, and Logistic Regression.')
    # st.image('./header.png')
    
    st.subheader('Select the Machine Learning Model :')
    model_selection = st.selectbox('Classification Models', ['SVM', 'Logistic', 'Perceptron'])
    
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    
    st.button('Make Prediction', on_click=make_prediction)
    st.checkbox('View Additional Information')
    
    if st.session_state.button_clicked:
        fig = make_prediction_and_plot(model_selection)
        st.success('The prediction is:')
        st.pyplot(fig)
        
if __name__ == '__main__':
    main()
