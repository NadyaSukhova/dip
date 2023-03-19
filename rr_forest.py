from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split #библиотека для машинного обучения
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json,io
from get_texts import get_links
from bs4 import BeautifulSoup
import requests

def rr_forest(texts):
    """
    links = get_links()
    text_list = []
    for link in list(links)[0:10]:
        source = requests.get(link)
        source.encoding = 'windows-1251'  # override encoding manually
        soup = BeautifulSoup(source.text, 'lxml')
        text = ""
        if soup.find("div", {"class": "full"}):
            for i in soup.find("div", {"class": "full"}).find_all("p", {"class": ""}):
                text += str(i.text)
        text_list.append(text)

    test_links = []
    for i in range(len(text_list)):
        link = {"text": text_list[i], "id": i}
        test_links.append(link)
    """
    # read date
    with io.open('input/train.json', encoding='utf-8') as f:
        raw_train = json.load(f)
    """
    with io.open('input/test.json', encoding='utf-8') as f:
        raw_test = json.load(f)
    """
    test_links = texts

    """test_links = raw_train[0:len(raw_train) // 3]
    raw_train = raw_train[len(raw_train) // 3 + 1:]"""
    def ru_token(string):
        """russian tokenize based on nltk.word_tokenize. only russian letter remaind."""
        return [i for i in word_tokenize(string) if re.match(r'[\u0400-\u04ffа́]+$', i)]

    params = {}
    params['tokenizer'] = ru_token
    params['stop_words'] = stopwords.words('russian')
    params['ngram_range'] = (1, 3)
    params['min_df'] = 3

    tfidf  = TfidfVectorizer(**params)

    tfidf_texts = [i['text'] for i in raw_train]
    for i in test_links:
        tfidf_texts += i
    tfidf.fit(tfidf_texts)

    train = {}
    val = {}
    tmp = defaultdict(list)
    for e in raw_train:
        tmp[e['sentiment']].append(e['text'])
    for l in tmp:
        train[l], val[l] = train_test_split(tmp[l], test_size=0.2, random_state=2018)

    def upsampling_align(some_dict, random_state=2018):
        rand = np.random.RandomState(random_state)
        upper = max([len(some_dict[l]) for l in some_dict])
        print('upper bound: {}'.format(upper))
        tmp = {}
        for l in some_dict:
            if len(some_dict[l]) < upper:
                repeat_time = int(upper/len(some_dict[l]))
                remainder = upper % len(some_dict[l])
                _tmp = some_dict[l].copy()
                rand.shuffle(_tmp)
                tmp[l] = some_dict[l] * repeat_time + _tmp[:remainder]
                rand.shuffle(tmp[l])
            else:
                tmp[l] = some_dict[l]
        return tmp

    """while n_estimators_max <=100:
        btrain = upsampling_align(train)
        #https://alexanderdyakonov.wordpress.com/2016/11/14/%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D1%8B%D0%B9-%D0%BB%D0%B5%D1%81-random-forest/
        m_params = {}
        m_params['n_estimators'] = 90
        m_params['max_depth'] = 70
        m_params['max_features'] = n_estimators_max
        m_params['criterion'] = 'gini'
        m_params['n_jobs'] = -1
    
        softmax = RandomForestClassifier(**m_params)
    
        train_x = [j for i in sorted(btrain.keys()) for j in btrain[i]]
        train_y = [i for i in sorted(btrain.keys()) for j in btrain[i]]
        softmax.fit(tfidf.transform(train_x), train_y)
    
    
        test_x = [j for i in sorted(val.keys()) for j in val[i]]
        true = [i for i in sorted(val.keys()) for j in val[i]]
    
        pred = softmax.predict(tfidf.transform(test_x))
    
        accuracy_score(true, pred)
    
        lab = LabelEncoder()
        c_true = lab.fit_transform(true)
        c_pred = lab.transform(pred)
    
    
        sub_pred = softmax.predict(tfidf.transform([i['text'] for i in test_links]))
        sub_df = pd.DataFrame()
        sub_df['id'] = [i['id'] for i in test_links]
        sub_df['true sent'] = [i['sentiment'] for i in test_links]
        sub_df['sentiment'] = sub_pred
    
        print(classification_report([i for i in sub_pred], [i['sentiment'] for i in test_links], target_names=['negative', 'neutral', 'positive'], digits=5))
    
        t = 0
        f = 0
        for i in range(len(sub_pred)):
            if sub_pred[i] == test_links[i]['sentiment']:
                t += 1
            f += 1
    
        print(t)
        print(f)
        print(t/f)
        print(n_estimators_max)
        if s < t/f:
            s = t/f
            n_estimators_res = n_estimators_max
        n_estimators_max+= 10
        sub_df.head()"""
    btrain = upsampling_align(train)
    # https://alexanderdyakonov.wordpress.com/2016/11/14/%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D1%8B%D0%B9-%D0%BB%D0%B5%D1%81-random-forest/
    m_params = {}
    m_params['n_estimators'] = 90
    m_params['max_depth'] = 70
    m_params['bootstrap'] = True
    m_params['criterion'] = 'gini'
    m_params['n_jobs'] = -1

    softmax = RandomForestClassifier(**m_params)

    train_x = [j for i in sorted(btrain.keys()) for j in btrain[i]]
    train_y = [i for i in sorted(btrain.keys()) for j in btrain[i]]
    softmax.fit(tfidf.transform(train_x), train_y)

    test_x = [j for i in sorted(val.keys()) for j in val[i]]
    true = [i for i in sorted(val.keys()) for j in val[i]]

    pred = softmax.predict(tfidf.transform(test_x))

    accuracy_score(true, pred)

    lab = LabelEncoder()
    c_true = lab.fit_transform(true)
    c_pred = lab.transform(pred)

    sub_pred = softmax.predict(tfidf.transform([i for i in test_links]))
    sub_df = pd.DataFrame()
    sub_df['sentiment'] = sub_pred
    sub_df['text'] = [i for i in test_links]

    sub_df.head()
    sub_df.to_csv('softmax_rr_forest.csv', index=False, encoding="utf-8-sig")
    return sub_df, classification_report(c_true, c_pred, target_names=lab.classes_, digits=5, output_dict=True)
#sub_df.to_csv('softmax_reg.csv', index=False)