import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#https://habr.com/ru/company/ruvds/blog/507778/
#streamlit run web-part.py
st.write("""
# Семантический анализ новостей
Данное приложение предскажет эмоциональную окраску новостей из указанного региона за указанный период времени
""")

st.sidebar.header('Выберите модели:')

def user_input_features():
    date1 = st.sidebar.date_input('Введите начало периода')
    date2 = st.sidebar.date_input('Введите конец периода')
    loc = st.sidebar.text_input('Введите название региона')
    log = st.sidebar.checkbox('Логистическая регресия')
    tree = st.sidebar.checkbox('Дерево решений')
    forest = st.sidebar.checkbox('Random forest')
    sgd = st.sidebar.checkbox('SGD')
    data = {'Logistic Regression': log,
            'DecisionTree': tree,
            'Random Forest': forest,
            'SGD': sgd}
    region = {'date1': date1,
            'date2': date2,
            'loc': loc}
    return data, region

df_model, df_region = user_input_features()

st.subheader('Введенне данные:')
st.write('Временной промежуток:')
st.write(df_region['date1'], ' - ', df_region['date2'])
st.write('Регион:')
st.write(df_region['loc'])
st.write(pd.DataFrame(df_model, index=[0]))
