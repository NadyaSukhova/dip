import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from get_texts import get_links
from rr_forest import rr_forest
from log_regres import log_regres
from tree import tree
from SGD import SGD

#https://habr.com/ru/company/ruvds/blog/507778/
#streamlit run web-part.py

st.write("""
# Семантический анализ новостей
""")

def user_input_features():
    st.sidebar.info('Формирование выборки')
    date1 = st.sidebar.date_input('Введите начало периода')
    date2 = st.sidebar.date_input('Введите конец периода')
    loc = st.sidebar.text_input('Введите название региона')
    st.sidebar.info('Выберите модели')
    log = st.sidebar.checkbox('Логистическая регресия')
    tree = st.sidebar.checkbox('Дерево решений')
    forest = st.sidebar.checkbox('Случайный лес')
    sgd = st.sidebar.checkbox('Стохастический градиентный спуск')
    data = {'Logistic Regression': log,
            'DecisionTree': tree,
            'Random Forest': forest,
            'SGD': sgd}
    region = {'date1': date1,
            'date2': date2,
            'loc': loc}
    return data, region

df_model, df_region = user_input_features()


methods = ''
if df_model['Logistic Regression'] == True:
    methods += 'Логистической регрессии'
if df_model['DecisionTree'] == True:
    if methods != '':
        methods += ', '
    methods += 'Дерева принятия решений'
if df_model['Random Forest'] == True:
    if methods != '':
        methods += ', '
    methods += 'Случайного леса'
if df_model['SGD'] == True:
    if methods != '':
        methods += ', '
    methods += 'Стохастического градиентного спуска'
if methods != '':
    st.subheader('Анализ будет проходить для методов ' + methods)
else:
    st.subheader('Выберите модели для анализа')

res_log, res_tree, res_forest, res_SGD = 0, 0, 0, 0

if 'Логистической регрессии' in methods:
    st.subheader('Метод логистической регрессии')
    df_log, res_log = log_regres(get_links(df_region['date1'], df_region['date2'], df_region['loc']))
    st.dataframe(df_log)
    st.text('Точность: ' + str(res_log['accuracy']))

if 'Дерева принятия решений' in methods:
    st.subheader('Метод дерева принятия решений')
    df_tree, res_tree = tree(get_links(df_region['date1'], df_region['date2'], df_region['loc']))
    st.dataframe(df_tree)
    st.text('Точность: ' + str(res_tree['accuracy']))

if 'Случайного леса' in methods:
    st.subheader('Метод случайного леса')
    df_forest, res_forest = rr_forest(get_links(df_region['date1'], df_region['date2'], df_region['loc']))
    st.dataframe(df_forest)
    st.text('Точность: ' + str(res_forest['accuracy']))

if 'Стохастического градиентного спуска' in methods:
    st.subheader('Метод стохастического градиентного спуска')
    df_SGD, res_SGD = SGD(get_links(df_region['date1'], df_region['date2'], df_region['loc']))
    st.dataframe(df_SGD)
    st.text('Точность: ' + str(res_SGD['accuracy']))

accur = dict()
if res_log != 0:
    accur['Логистическая регресия'] = [res_log['accuracy']]
if res_tree != 0:
    accur['Дерево решений'] = [res_tree['accuracy']]
if res_forest != 0:
    accur['Случайный лес'] = [res_forest['accuracy']]
if res_SGD != 0:
    accur['Стохастический градиентный спуск'] = [res_SGD['accuracy']]

if accur != {}:
    st.subheader('Таблица точности методов')
    st.dataframe(pd.DataFrame(accur))
    st.subheader('Вывод')
    best_accur = max(accur.values())[0]
    best_accur_name = max(accur, key=lambda x: accur[x])
    if best_accur_name == 'Логистическая регресия':
        df_res = df_log
    if best_accur_name == 'Дерево решений':
        df_res = df_tree
    if best_accur_name == 'Случайный лес':
        df_res = df_forest
    if best_accur_name == 'Стохастический градиентный спуск':
        df_res = df_SGD

    st.write("Максимальная точность " + str(best_accur) + " была получена методом \"" + best_accur_name + "\"")
    sent_count = Counter(df_res.to_dict('dict')['sentiment'].values())
    st.subheader('Гистограмма')
    chart_data = pd.DataFrame(
        [[sent_count['positive'],0,0], [0,sent_count['neutral'],0], [0,0,sent_count['negative']]],
        columns=["positive", "neutral", "negative"])

    st.bar_chart(chart_data)
