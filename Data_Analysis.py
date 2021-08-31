# Представлены данные такси, по которым мы должны предсказать возьмёт ли таксист заказ или нет(driver_response).
#
# Чек-лист:
#
# 1.Загрузите датасет taxi.csv.
# 2.Посмотрите на данные. Отобразите общую информацию по признакам (вспомните о describe и info). Напишите в markdown свои наблюдения.
# 3.Выявите пропуски, а также возможные причины их возникновения. Решите, что следует сделать с ними. Напишите в markdown свои наблюдения.
# 4.Оцените зависимости переменных между собой. Используйте корреляции. Будет хорошо, если воспользуетесь profile_report. Напишите в markdown свои наблюдения.
# 5.Определите стратегию преобразования категориальных признаков (т.е. как их сделать адекватными для моделей).
# 6.Найдите признаки, которые можно разделить на другие, или преобразовать в другой тип данных. Удалите лишние, при необходимости.
# 7.Разделите выборку на обучаемую и тестовую.
# 8.Обучите модель. Напишите в markdown свои наблюдения по полученным результатам. Хорошие результаты дают классификаторы RandomForest и XGBoost

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
#import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# Сброс ограничений на число столбцов
pd.set_option('display.max_columns', None)

data = pd.read_csv('taxi.csv')
print(data.head(10))

# Проверим, нет ли в данных пропущенных значений.
print(data.info())
print(data.isna().sum())

# Пропущенных значений нет!

# Проверим данные на аномалии и выбросы
print(data.describe())

# данные почищены. Пропущенные значения возможно заменены на -1.
# Например, в колонках origin_order_latitude, origin_order_longitude, distance_km,duration_min. Есть выбросы на очень большое расстояние и соответственно длительность поездки.

# Удаление дубликатов
data.drop_duplicates()

# Дубликатов не было.

feature_names = data.columns.tolist() 
for column in feature_names: 
    print(column)
    print(data[column].value_counts(dropna=False))
    
#Выяснил, что параметры дистанция и длительность поездки для 26207 значений = -1, и для 152 значений = 0. Это чуть боьше четверти всех данных. 
#Так же координаты широты и долготы для заказов origin_order_latitude, origin_order_longitude совпалают для многих тысяч значений. 
#Скорее всего заказы поступают из одних и тех же домов.

#Визуально сравним для примера взаимосвязь дистанции и длительности поездки, чтобы выявить возможные аномалии.

plt.figure(figsize=(12,6))
plt.plot(data.distance_km)
plt.plot(data.duration_min)
plt.show()

# Видно чёткую зависимость. Визуально аномалий не наблюдается.
# Плотности распределения всех параметров исходного датасета data
plt.figure(figsize=(10, 16))
sns.pairplot(data)
plt.show()

# Создадим второй датасет отфильтрованный по дистанции поездки. Выкинем отрицательные, нулевые и очень большие значения.
data1 = data.query('distance_km > 0 and distance_km < 200')
print(data1.describe())

# Делаем переиндексацию после удаления строк из исходного датафрейма
data1.reset_index(inplace=True)
data1.index.max()

# Частотный график дистанции поездки
ax = plt.figure(figsize=(14,6))
plt.hist(data1['distance_km'])
plt.show()

# Плотности распределения всех параметров очищенного датасета data1
plt.figure(figsize=(10, 16))
sns.pairplot(data1)
plt.show()

# Оцените зависимости переменных между собой. Используйте корреляции. Будет хорошо, если воспользуетесь profile_report. Напишите в markdown свои наблюдения.
# Построим матрицу корреляций и оценим взаимную зависимость признаков:
plt.figure(figsize=(13, 8))
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
plt.show()

# Из этой матрицы можно сделать выводы: Коррелируют между собой признаки широты и долготы, а так же дистанции и времени поездки. Это вполне ожидаемо.
plt.figure(figsize=(13, 8))
sns.heatmap(data1.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
plt.show()

#Значения корреляции для нового урезанного датасета изменились незначительно. Обратнопророрциональная зависимость между driver_response и (distance_km с duration_min) стала более выраженной.

#Определите стратегию преобразования категориальных признаков (т.е. как их сделать адекватными для моделей).
# выделяем данные определённого типа - категориальные переменные
obj_data = data1.select_dtypes(include=['object']).copy()
print(obj_data.head())

feature_names = data1.select_dtypes(include=['object']).copy()
for column in feature_names: 
    print(column)
    print(data1[column].value_counts(dropna=False))

# Преобразуем категориальные переменные

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
label_enc = LabelEncoder()
offer_class_label = label_enc.fit_transform(data1['offer_class_group'])
data1.loc[:, 'offer_class_group'] = offer_class_label
ride_type_desc = label_enc.fit_transform(data1['ride_type_desc'])
data1.loc[:, 'ride_type_desc'] = ride_type_desc

feature_names = data1[['offer_class_group','ride_type_desc']].columns.tolist() 
for column in feature_names: 
    print(column)
    print(data1[column].value_counts(dropna=False))

# Находим признаки, которые можно разделить на другие, или преобразовать в другой тип данных. Удалите лишние, при необходимости.
# data1.columns
#
# Между собой коррелируют признаки:

# 'distance_km', 'duration_min'
# 'driver_latitude', 'driver_longitude'
# 'origin_order_latitude', 'origin_order_longitude'
#
# Заменим каждую пару на составной признак

import reverse_geocoder as rg
import pprint as pp

# Функция определяет местоположение по координатам для датасета данных с построчным извлечением координат
# def reverseGeocode(dataset):
#     result = rg.search((dataset['origin_order_latitude'],dataset['origin_order_longitude']))[0]['name']
#     return result

#Пример применения функции

# data2 = data1[['origin_order_latitude','origin_order_longitude']].head(100)
# data2['conversion'] = data2.apply(reverseGeocode, axis=1)
# print(data2)

# Функция работает очень медленно, поэтому решил применить другой способ обработки координат
data1.loc[:,'driver_coordinates'] = data1['driver_latitude'] * data1['driver_longitude'] 
data1.loc[:,'order_coordinates'] = data1['origin_order_latitude'] * data1['origin_order_longitude'] 
data1.loc[:,'km_minut'] = data1['duration_min'] / data1['distance_km'] 
print(data1.head())

# Удалим лишние признаки
data1 = data1.drop(['index', 'offer_gk','driver_latitude', 'driver_longitude', 'origin_order_latitude',
       'origin_order_longitude', 'distance_km', 'duration_min'], axis = 1)
data_col1 = ['weekday_key', 'hour_key', 'offer_class_group', 'ride_type_desc', 'driver_response'
             , 'km_minut','driver_coordinates', 'order_coordinates']
             
data_colX1 = ['weekday_key', 'hour_key', 'driver_gk', 'order_gk', 'offer_class_group',
       'ride_type_desc', 'km_minut', 'driver_coordinates','order_coordinates']
       
data_colX2 = ['km_minut']
dataX = data1[data_colX1]

# Разделим выборку на обучаемую и тестовую
output_y = data1[['driver_response']]
input_x = data1[data_colX1]
from sklearn.model_selection import train_test_split
X_train_22, X_test_22, y_train_22, y_test_22 = train_test_split(input_x, output_y, test_size=0.2)

#Обучите модель. Напишите в markdown свои наблюдения по полученным результатам. Хорошие результаты дают классификаторы RandomForest и XGBoost
from sklearn.ensemble import RandomForestClassifier

# создаем модель деревья решений
# выбираем 100 деревьев в качестве параметра с глубиной дерева 20
model=RandomForestClassifier(n_estimators=100, max_depth=20)
# обучаем модель
model.fit(X_train_22,y_train_22)
import sys
RandomForestClassifier(max_depth=20)

#Наиболее важные атрибуты модели RandomForest
headers = list(X_train_22.columns.values)
feature_imp = pd.Series(model.feature_importances_,index=headers).sort_values(ascending=False)
f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Важность атрибутов')
plt.ylabel('Атрибуты')
plt.title("Наиболее важные атрибуты")
#plt.legend()
plt.show()

#Можно сделать вывод, что наиболее значимые атрибуты это дальность поездки (у нас это время затраченное на километр), координаты водителя и координаты заказа.

#Посчитаем качество модели - classification report:
from sklearn.metrics import classification_report
model_pred = model.predict(X_test_22)
print(classification_report(y_test_22, model_pred))
print('Точность предсказания модели на тестовых данных:', model.score(X_test_22, y_test_22))
print('Точность предсказания модели на тренировочных данных:',model.score(X_train_22, y_train_22))

#Точность предсказания модели на тестовых данных: 0.7975397580535545
#Точность предсказания модели на тренировочных данных: 0.9672081011281772
#Результаты точности предсказания при разных параметрах RandomForest: Модель "RandomForest" с глубиной дерева 5 дала точность предсказания 74.2%. Модель "RandomForest" с глубиной дерева 10 дала точность предсказания 78%. Модель "RandomForest" с глубиной дерева 20 дала точность предсказания 80%. При глубине дерева более 15 происходит переобучение модели. Разрыв точности между тестом и трейном становится большой. Количество деревьев, начиная с 50 не влияет на точность данной модели.

#Проверим какая будет точность модели, если оставить только один самый значимый параметр "km_minut"
# Разделим выборку на обучаемую и тестовую
output_y = data1[['driver_response']]
input_x = data1[data_colX2]
from sklearn.model_selection import train_test_split
X_train_22, X_test_22, y_train_22, y_test_22 = train_test_split(input_x, output_y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

# создаем модель деревья решений
# выбираем 100 деревьев в качестве параметра с глубиной дерева 20
model=RandomForestClassifier(n_estimators=100, max_depth=20)

# обучаем модель
model.fit(X_train_22,y_train_22)
import sys
RandomForestClassifier(max_depth=20)
#Посчитаем качество модели - classification report:
from sklearn.metrics import classification_report
model_pred = model.predict(X_test_22)
print(classification_report(y_test_22, model_pred))
print('Точность предсказания модели на тестовых данных:', model.score(X_test_22, y_test_22))
print('Точность предсказания модели на тренировочных данных:',model.score(X_train_22, y_train_22))

#Точность предсказания модели на тестовых данных: 0.6937610439037651
#Точность предсказания модели на тренировочных данных: 0.7739907571020797
#Точность предсказания для модели с одним параметром значительно хуже - 69.4%.

#Применим метод Градиентный бустинг XGBoost для предсказательной модели

#!pip install xgboost
import xgboost
from sklearn.metrics import accuracy_score

# Разделим выборку на обучаемую и тестовую
output_y = data1[['driver_response']]
input_x = data1[data_colX1]

from sklearn.model_selection import train_test_split
X_train_22, X_test_22, y_train_22, y_test_22 = train_test_split(input_x, output_y, test_size=0.2)
#По умолчанию, предсказания, сделанные XGBoost являются вероятностями. 
#Поскольку это бинарная задача классификации, каждое предсказание является вероятностью принадлежности к первому классу. 
#Поэтому мы можем легко преобразовать их в значения двоичных классов путем округления до 0 или 1.

# fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train_22, y_train_22)

# make predictions for test data
y_pred = model.predict(X_test_22)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test_22, predictions)
print("Точность предсказания модели : %.2f%%" % (accuracy * 100.0))

# Точность предсказания модели: 82.76%
# Точность предсказания модели XGBoost (82.8%) получилась чуть больше точности предсказания модели RandomForest (80%).