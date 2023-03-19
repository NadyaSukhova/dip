from bs4 import BeautifulSoup
import requests
from datetime import datetime
from datetime import timedelta


def get_links(date1, date2, loc):
# Получаем дату начала в формате datetime
    date_obj_1 = date1
    # Получаем дату окончания в формате datetime
    date_obj_2 = date2
    # Объявляем переменную, в которой будут храниться ссылки
    links = []
    while date_obj_2 >= date_obj_1:
        source = requests.get('https://ria.ru/location_' + loc + '/' + str(date1).replace('-', ''))
        source.encoding = 'windows-1251'
        soup = BeautifulSoup(source.text, 'lxml')
        try:
            # В указанном классе хранятся все необходимые ссылки, которые сохраняются
            for i in soup.find("div", {"class": 'rubric-list'}).find_all("a", {"class": "list-item__title color-font-hover-only"}):
                link_a = i
                link = link_a.get('href')
                links.append(link)
        except:
            pass

        date_obj_1 = date_obj_1 + timedelta(days=1)
        date1 = date_obj_1.strftime('%Y%m%d')
    texts = []
    for link in links:
        source = requests.get(link)
        soup = BeautifulSoup(source.text, 'lxml')
        text = ''
        if soup.find("div", {"class": 'article__body js-mediator-article mia-analytics'}):
            for i in soup.find("div", {"class": 'article__body js-mediator-article mia-analytics'}):
                text += i.get_text() + ' '
            texts.append(text)

    return texts
