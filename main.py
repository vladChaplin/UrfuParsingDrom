import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def main(nameCity):

    global response
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    cars = []
    for i in range(85):
        try:
            if nameCity == 'Екатеринбург':
                response = requests.get(f"https://ekaterinburg.drom.ru/chevrolet/cruze/page{i}/?maxyear=2019&ph=1&unsold=1", headers=headers)
            #response = requests.get(f"https://auto.drom.ru/chevrolet/cruze/page{i}/?maxyear=2019&ph=1&unsold=1",                                    headers=headers)
            # print(response.url)
            response.raise_for_status()  # проверка успешности запроса
            soup = BeautifulSoup(response.content, "html.parser")
            tags = soup.find_all("div", {"data-ftid": "bulls-list_bull"})

            if not tags:
                print(f"No tags found on page {i}")
                continue

            for tag in tags:
                try:
                    title_tag = tag.find("a", {"data-ftid": "bull_title"})
                    name_year = title_tag.text.strip() if title_tag else ""
                    name, year = name_year.split(',') if ',' in name_year else (name_year, "")

                    price_tag = tag.find("span", {"data-ftid": "bull_price"})
                    price = price_tag.text.replace('\xa0', '').replace('₽', '').strip() if price_tag else ""

                    city_tag = tag.find("span", {"data-ftid": "bull_location"})
                    city = city_tag.text.strip() if city_tag else ""

                    if price and name and city:  # Проверка, что данные не пустые
                        cars.append({"car": name.strip(), "year": year.strip(), "price": float(price.replace(' ', '')),
                                     "city": city.strip()})

                except Exception as e:
                    print(f"Error processing tag: {e}")

            # Задержки между запросами для избежания блокировки со стороны сервера
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    print(cars)

    # Преобразуем список словарей в DataFrame для дальнейшего анализа
    df = pd.DataFrame(cars)
    median = df.price.median()
    df.year = df.year.astype(int)
    print(df.price)

    # Распределение цены
    q = np.quantile(df.price, 0.99)
    df.loc[df.price > q, 'price'] = q
    plt.hist(df.price, bins=30)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Price Distribution of Chevrolet Cruze Cars')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df[df.city == nameCity], x='year', y='price')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title('Price vs Year for Chevrolet Cruze Cars in ' + nameCity)
    plt.show()


if __name__ == '__main__':
    main('Екатеринбург')

