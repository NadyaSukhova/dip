from bs4 import BeautifulSoup
import requests
from datetime import datetime
from datetime import timedelta


def get_links():
# Получаем дату начала в формате datetime
    d1 = input("Введите дату \"с\" в формате дд.мм.гггг:")
    date1 = d1[6::] + "/" + d1[3:5] + "/" + d1[:2]
    date_obj_1 = datetime.strptime(date1, '%Y/%m/%d')
    # Получаем дату окончания в формате datetime
    d2 = input("Введите дату \"по\" в формате дд.мм.гггг:")
    date2 = d2[6::] + "/" + d2[3:5] + "/" + d2[:2]
    date_obj_2 = datetime.strptime(date2, '%Y/%m/%d')
    # Объявляем переменную, в которой будут храниться ссылки
    links = []
    while date_obj_2 >= date_obj_1:
        source = requests.get('https://www.vzsar.ru/news/date/' + date1)
        source.encoding = 'windows-1251'
        soup = BeautifulSoup(source.text, 'lxml')
        # В указанном классе хранятся все необходимые ссылки, которые сохраняются
        for i in soup.find("div", {"class": "newslist"}).find_all("a"):
            link_a = i
        link = link_a.get('href')
        links.append('https://www.vzsar.ru' + link)

        date_obj_1 = date_obj_1 + timedelta(days=1)
        date1 = date_obj_1.strftime('%Y/%m/%d')
    return links
