# import io
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import seaborn as sns
#
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.linear_model import SGDRegressor
# from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
# from sklearn.metrics import mean_squared_error as mse, r2_score
# from sklearn.metrics import PredictionErrorDisplay
# import warnings
#
# warnings.filterwarnings('ignore')
# pd.options.display.max_columns = None
#
# translations = {
#     'year': 'год',
#     'distance(km)': 'пробег (км)',
#     'engine_capacity(L)': 'объем двигателя (л)',
#     'fuel_type_бензин': 'тип топлива - бензин',
#     'fuel_type_дизель': 'тип топлива - дизель',
#     'fuel_type_электро': 'тип топлива - электро',
#     'fuel_type_ГБО': 'тип топлива - ГБО',
#     'fuel_type_гибрид': 'тип топлива - гибрид'
# }
#
# # Класс для замены выбросов
# class QuantileReplacer(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold=0.05):
#         self.threshold = threshold
#         self.quantiles = {}
#
#     def fit(self, X, y=None):
#         for col in X.select_dtypes(include='number'):
#             low_quantile = X[col].quantile(self.threshold)
#             high_quantile = X[col].quantile(1 - self.threshold)
#             self.quantiles[col] = (low_quantile, high_quantile)
#         return self
#
#     def transform(self, X):
#         X_copy = X.copy()
#         for col in X.select_dtypes(include='number'):
#             low_quantile, high_quantile = self.quantiles[col]
#             rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
#             if rare_mask.any():
#                 rare_values = X_copy.loc[rare_mask, col]
#                 replace_value = np.mean([low_quantile, high_quantile])
#                 if rare_values.mean() > replace_value:
#                     X_copy.loc[rare_mask, col] = high_quantile
#                 else:
#                     X_copy.loc[rare_mask, col] = low_quantile
#         return X_copy
#
#
# def main(cities, selected_brands, selected_models, years):
#     # Проверка выбора значений из выпадающих списков
#     if not cities:
#         st.write("Пожалуйста, выберите город.")
#         return
#
#     if not selected_brands:
#         st.write("Пожалуйста, выберите марку автомобиля.")
#         return
#
#     if not any(selected_models.values()):
#         st.write("Пожалуйста, выберите модель автомобиля.")
#         return
#
#     # Загрузка данных на основе выбора пользователя
#     df = pd.DataFrame()
#     if 'Екатеринбург' in cities:
#         if 'Hyundai' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_solaris.csv')
#         if 'Toyota' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_camry.csv')
#         if 'Lada' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_lada.csv')
#     if 'Москва' in cities:
#         if 'Hyundai' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_solaris.csv')
#         if 'Toyota' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_camry.csv')
#         if 'Lada' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_lada.csv')
#
#     if df.empty:
#         st.write("Нет данных для выбранных условий.")
#         return
#
#     df = df.dropna()
#
#     st.title('Анализ данных об автомобилях из города ')
#
#     st.subheader('Информация о данных')
#     # buffer = io.StringIO()
#     # df.info(buf=buffer)
#     # s = buffer.getvalue()
#     # st.text(s)
#
#     # Определяем числовые и категориальные признаки
#     num = ['year', 'distance(km)', 'engine_capacity(L)']
#     cat = ['fuel_type']
#
#     # Pipeline для числовых признаков
#     num_pipe = Pipeline([
#         ('QuantReplace', QuantileReplacer(threshold=0.01)),
#         ('norm', MinMaxScaler())
#     ])
#
#     # Pipeline для категориальных признаков
#     cat_pipe = Pipeline([
#         ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
#     ])
#
#     # Создаем ColumnTransformer для всех признаков
#     preprocessors_All = ColumnTransformer(transformers=[
#         ('num', num_pipe, num),
#         ('cat', cat_pipe, cat)
#     ])
#
#     # Удаляем целевую переменную из признаков
#     X, y = df.drop(columns=['price(rub)']), df['price(rub)']
#
#     # Преобразуем признаки
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_train_prep = preprocessors_All.fit_transform(X_train)
#     X_val_prep = preprocessors_All.transform(X_val)
#
#     # Проверка, что количество признаков совпадает
#     if X_train_prep.shape[1] != len(num) + len(cat):
#         st.write("Количество признаков после трансформации не совпадает с ожидаемым. Проверьте данные.")
#         return
#
#     # Обучаем модель
#     model = SGDRegressor(random_state=42)
#     model.fit(X_train_prep, y_train)
#
#     # Функция для получения коэффициентов модели
#     def get_coefs(model):
#         B0 = model.intercept_[0]
#         B = model.coef_
#         return B0, B
#
#     # Функция для отображения уравнения модели с пояснениями
#     def print_model(B0, B, features_names):
#         line = '{:.3f}'.format(B0)
#         sign = ['+', '-']
#         for p, (fn, b) in enumerate(zip(features_names, B)):
#             line = line + sign[int(0.5 * (np.sign(b) - 1))] + '{:.2f}*'.format(np.abs(b)) + fn
#         st.write('Уравнение линейной регрессии:')
#         st.write(f'Цена (руб) = {line}')
#         st.write('Где:')
#         for fn in features_names:
#             st.write(f'- {fn} = {translations.get(fn, fn)}')
#
#     # Функция для визуализации весов модели
#     def vis_weigths(weights, features_names=None, width=1200, height=600):
#         numbers = np.arange(0, len(weights))
#         if features_names:
#             tick_labels = np.hstack(['B0', features_names])
#         else:
#             tick_labels = ['B' + str(num) for num in numbers]
#         fig = go.Figure()
#         fig.add_trace(
#             go.Bar(x=numbers[weights < 0], y=weights[weights < 0], marker_color='red', name='Отрицательные веса'))
#         fig.add_trace(
#             go.Bar(x=numbers[weights >= 0], y=weights[weights >= 0], marker_color='blue', name='Положительные веса'))
#         fig.update_layout(
#             title="Веса модели",
#             width=width,
#             height=height,
#             template="plotly_dark",
#             xaxis=dict(tickmode='array', tickvals=numbers, ticktext=tick_labels)
#         )
#         st.plotly_chart(fig)
#
#     # Получаем названия признаков после OneHotEncoding
#     cat_features = preprocessors_All.transformers_[1][1]['encoder'].get_feature_names_out(cat)
#     features_names = list(num) + list(cat_features)
#
#     # Получаем коэффициенты модели и отображаем уравнение
#     B0, B = get_coefs(model)
#     print_model(B0, B, features_names)
#
#     # Визуализируем веса модели
#     Bs = np.hstack([B0, B])
#     vis_weigths(Bs, features_names)
#
#     # Функция для расчета метрики
#     def calculate_metric(model_pipe, X, y, metric=r2_score, **kwargs):
#         y_model = model_pipe.predict(X)
#         return metric(y, y_model, **kwargs)
#
#     # Отображаем метрики модели
#     st.write(f"R2 на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train):.4f}")
#     st.write(f"R2 на валидационной выборке: {calculate_metric(model, X_val_prep, y_val):.4f}")
#
#     st.write(f"MSE на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train, mse):.4f}")
#     st.write(f"MSE на валидационной выборке: {calculate_metric(model, X_val_prep, y_val, mse):.4f}")
#
#     st.write(f"RMSE на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train, mse, squared=False):.4f}")
#     st.write(f"RMSE на валидационной выборке: {calculate_metric(model, X_val_prep, y_val, mse, squared=False):.4f}")
#
#     # Кросс-валидация
#     def cross_validation(X, y, model, scoring, cv_rule):
#         scores = cross_validate(model, X, y, scoring=scoring, cv=cv_rule)
#         st.write('Ошибка на кросс-валидации')
#         DF_score = pd.DataFrame(scores)
#         st.write(DF_score)
#         st.write('\n')
#         st.write(DF_score.mean()[2:])
#
#     scoring_reg = {'R2': 'r2', '-MSE': 'neg_mean_squared_error', '-MAE': 'neg_mean_absolute_error', '-Max': 'max_error'}
#     cross_validation(X_train_prep, y_train, model, scoring_reg, ShuffleSplit(n_splits=5, random_state=42))
#
#     # Визуализация ошибок предсказания
#     fig, ax = plt.subplots()
#     PredictionErrorDisplay.from_predictions(
#         y_val, model.predict(X_val_prep), kind="actual_vs_predicted",
#         scatter_kwargs={"alpha": 0.5}, ax=ax
#     )
#     ax.set_xlabel("Предсказанные значения (руб)")
#     ax.set_ylabel("Фактические значения (руб)")
#     st.pyplot(fig)
#
#     # Построение итогового графика линейной регрессии
#     def plot_regression_results(model, X, y, feature_names):
#         fig, axes = plt.subplots(1, len(feature_names), figsize=(15, 5))
#         for i, ax in enumerate(axes):
#             sns.regplot(x=X[:, i], y=y, ax=ax, line_kws={"color": "red"})
#             ax.set_xlabel(translations.get(feature_names[i], feature_names[i]))
#             ax.set_ylabel("Цена (руб)")
#         st.pyplot(fig)
#
#     plot_regression_results(model, X_val_prep, y_val, features_names)
#
#
# # Streamlit UI
# st.title('Анализ цен на автомобили')
# st.write('### Линейная регрессия цен на автомобили по годам')
#
# # Выбор городов
# cities = st.multiselect('Выберите города', ['Екатеринбург', 'Москва'])
#
# # Словарь марок и моделей автомобилей
# model_dict = {
#     'Hyundai': ['Solaris'],
#     'Toyota': ['Camry'],
#     'Lada': ['Granta']
# }
#
# # Выбор марок автомобилей
# selected_brands = st.multiselect('Выберите марки автомобилей', list(model_dict.keys()))
#
# # Фильтрация моделей по выбранным маркам
# selected_models = {brand: st.multiselect(f'Выберите модели для {brand}', model_dict[brand]) for brand in selected_brands}
#
# # Выбор диапазона лет
# years = st.slider('Года', 2008, 2019, (2008, 2019))
#
# if st.button('Выполнить'):
#     main(cities, selected_brands, selected_models, years)
#
# if st.button('Очистить'):
#     st.cache_data.clear()
#     st.experimental_rerun()

#===================================

# import io
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import seaborn as sns
#
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.linear_model import SGDRegressor
# from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
# from sklearn.metrics import mean_squared_error as mse, r2_score
# from sklearn.metrics import PredictionErrorDisplay
# import warnings
#
# warnings.filterwarnings('ignore')
# pd.options.display.max_columns = None
#
# translations = {
#     'year': 'год',
#     'distance(km)': 'пробег (км)',
#     'engine_capacity(L)': 'объем двигателя (л)',
#     'fuel_type_бензин': 'тип топлива - бензин',
#     'fuel_type_дизель': 'тип топлива - дизель',
#     'fuel_type_электро': 'тип топлива - электро',
#     'fuel_type_ГБО': 'тип топлива - ГБО',
#     'fuel_type_гибрид': 'тип топлива - гибрид'
# }
#
# class QuantileReplacer(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold=0.05):
#         self.threshold = threshold
#         self.quantiles = {}
#
#     def fit(self, X, y=None):
#         for col in X.select_dtypes(include='number'):
#             low_quantile = X[col].quantile(self.threshold)
#             high_quantile = X[col].quantile(1 - self.threshold)
#             self.quantiles[col] = (low_quantile, high_quantile)
#         return self
#
#     def transform(self, X):
#         X_copy = X.copy()
#         for col in X.select_dtypes(include='number'):
#             low_quantile, high_quantile = self.quantiles[col]
#             rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
#             if rare_mask.any():
#                 rare_values = X_copy.loc[rare_mask, col]
#                 replace_value = np.mean([low_quantile, high_quantile])
#                 if rare_values.mean() > replace_value:
#                     X_copy.loc[rare_mask, col] = high_quantile
#                 else:
#                     X_copy.loc[rare_mask, col] = low_quantile
#         return X_copy
#
# def main(cities, selected_brands, selected_models, years):
#     if not cities:
#         st.write("Пожалуйста, выберите город.")
#         return
#
#     if not selected_brands:
#         st.write("Пожалуйста, выберите марку автомобиля.")
#         return
#
#     if not any(selected_models.values()):
#         st.write("Пожалуйста, выберите модель автомобиля.")
#         return
#
#     df = pd.DataFrame()
#     if 'Екатеринбург' in cities:
#         if 'Hyundai' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_solaris.csv')
#         if 'Toyota' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_camry.csv')
#         if 'Lada' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_lada.csv')
#     if 'Москва' in cities:
#         if 'Hyundai' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_solaris.csv')
#         if 'Toyota' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_camry.csv')
#         if 'Lada' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_granta.csv')
#
#     if df.empty:
#         st.write("Нет данных для выбранных условий.")
#         return
#
#     df = df.dropna()
#
#     st.title(f"Анализ данных об автомобилях из города {' и '.join(cities)} и марки {', '.join(selected_brands)}")
#
#     st.subheader('Информация о данных')
#
#     num = ['year', 'distance(km)', 'engine_capacity(L)']
#     cat = ['fuel_type']
#
#     num_pipe = Pipeline([
#         ('QuantReplace', QuantileReplacer(threshold=0.01)),
#         ('norm', MinMaxScaler())
#     ])
#
#     cat_pipe = Pipeline([
#         ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
#     ])
#
#     preprocessors_All = ColumnTransformer(transformers=[
#         ('num', num_pipe, num),
#         ('cat', cat_pipe, cat)
#     ])
#
#     X, y = df.drop(columns=['price(rub)']), df['price(rub)']
#
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_train_prep = preprocessors_All.fit_transform(X_train)
#     X_val_prep = preprocessors_All.transform(X_val)
#
#     if X_train_prep.shape[1] != len(num) + len(cat):
#         st.write("Количество признаков после трансформации не совпадает с ожидаемым. Проверьте данные.")
#         return
#
#     model = SGDRegressor(random_state=42)
#     model.fit(X_train_prep, y_train)
#
#     def get_coefs(model):
#         B0 = model.intercept_[0]
#         B = model.coef_
#         return B0, B
#
#     def print_model(B0, B, features_names):
#         line = '{:.3f}'.format(B0)
#         sign = ['+', '-']
#         for p, (fn, b) in enumerate(zip(features_names, B)):
#             line = line + sign[int(0.5 * (np.sign(b) - 1))] + '{:.2f}*'.format(np.abs(b)) + fn
#         st.write('Уравнение линейной регрессии:')
#         st.write(f'Цена (руб) = {line}')
#         st.write('Где:')
#         for fn in features_names:
#             st.write(f'- {fn} = {translations.get(fn, fn)}')
#
#     def vis_weigths(weights, features_names=None, width=1200, height=600):
#         numbers = np.arange(0, len(weights))
#         if features_names:
#             tick_labels = np.hstack(['B0', features_names])
#         else:
#             tick_labels = ['B' + str(num) for num in numbers]
#         fig = go.Figure()
#         fig.add_trace(
#             go.Bar(x=numbers[weights < 0], y=weights[weights < 0], marker_color='red', name='Отрицательные веса'))
#         fig.add_trace(
#             go.Bar(x=numbers[weights >= 0], y=weights[weights >= 0], marker_color='blue', name='Положительные веса'))
#         fig.update_layout(
#             title="Веса модели",
#             width=width,
#             height=height,
#             template="plotly_dark",
#             xaxis=dict(tickmode='array', tickvals=numbers, ticktext=tick_labels)
#         )
#         st.plotly_chart(fig)
#
#     cat_features = preprocessors_All.transformers_[1][1]['encoder'].get_feature_names_out(cat)
#     features_names = list(num) + list(cat_features)
#
#     B0, B = get_coefs(model)
#     print_model(B0, B, features_names)
#
#     Bs = np.hstack([B0, B])
#     vis_weigths(Bs, features_names)
#
#     def calculate_metric(model_pipe, X, y, metric=r2_score, **kwargs):
#         y_model = model_pipe.predict(X)
#         return metric(y, y_model, **kwargs)
#
#     st.write(f"R2 на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train):.4f}")
#     st.write(f"R2 на валидационной выборке: {calculate_metric(model, X_val_prep, y_val):.4f}")
#
#     st.write(f"MSE на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train, mse):.4f}")
#     st.write(f"MSE на валидационной выборке: {calculate_metric(model, X_val_prep, y_val, mse):.4f}")
#
#     st.write(f"RMSE на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train, mse, squared=False):.4f}")
#     st.write(f"RMSE на валидационной выборке: {calculate_metric(model, X_val_prep, y_val, mse, squared=False):.4f}")
#
#     def cross_validation(X, y, model, scoring, cv_rule):
#         scores = cross_validate(model, X, y, scoring=scoring, cv=cv_rule)
#         st.write('Ошибка на кросс-валидации')
#         DF_score = pd.DataFrame(scores)
#         st.write(DF_score)
#         st.write('\n')
#         st.write(DF_score.mean()[2:])
#
#     scoring_reg = {'R2': 'r2', '-MSE': 'neg_mean_squared_error', '-MAE': 'neg_mean_absolute_error', '-Max': 'max_error'}
#     cross_validation(X_train_prep, y_train, model, scoring_reg, ShuffleSplit(n_splits=5, random_state=42))
#
#     fig, ax = plt.subplots()
#     PredictionErrorDisplay.from_predictions(
#         y_val, model.predict(X_val_prep), kind="actual_vs_predicted",
#         scatter_kwargs={"alpha": 0.5}, ax=ax
#     )
#     ax.set_xlabel("Предсказанные значения (руб)")
#     ax.set_ylabel("Фактические значения (руб)")
#     st.pyplot(fig)
#
#     def plot_regression_results(model, X, y, feature_names):
#         fig, axes = plt.subplots(1, len(feature_names), figsize=(15, 5))
#         for i, ax in enumerate(axes):
#             sns.regplot(x=X[:, i], y=y, ax=ax, line_kws={"color": "red"})
#             ax.set_xlabel(translations.get(feature_names[i], feature_names[i]))
#             ax.set_ylabel("Цена (руб)")
#         st.pyplot(fig)
#
#     plot_regression_results(model, X_val_prep, y_val, features_names)
#
#     def predict_next_year(model, X, features_names):
#         X_copy = X.copy()
#         if 'year' in features_names:
#             year_index = features_names.index('year')
#             X_copy[:, year_index] += 1
#             y_pred = model.predict(X_copy)
#             return y_pred
#         else:
#             st.write("Признак 'year' отсутствует в данных.")
#             return None
#
#     next_year_predictions = predict_next_year(model, X_val_prep, features_names)
#     if next_year_predictions is not None:
#         max_price = next_year_predictions.max()
#         mean_price = next_year_predictions.mean()
#         min_price = next_year_predictions.min()
#
#         st.write("### Прогнозируемая стоимость автомобилей на следующий год:")
#         st.markdown(f"<h3 style='color: #33BCFF;'>Максимальная: {max_price:.2f} руб.</h3>", unsafe_allow_html=True)
#         st.markdown(f"<h3 style='color: #33BCFF;'>Средняя: {mean_price:.2f} руб.</h3>", unsafe_allow_html=True)
#         st.markdown(f"<h3 style='color: #33BCFF;'>Минимальная: {min_price:.2f} руб.</h3>", unsafe_allow_html=True)
#
# st.title('Анализ цен на автомобили')
# st.write('### Линейная регрессия цен на автомобили по годам')
#
# cities = st.multiselect('Выберите города', ['Екатеринбург', 'Москва'])
#
# model_dict = {
#     'Hyundai': ['Solaris'],
#     'Toyota': ['Camry'],
#     'Lada': ['Granta']
# }
#
# selected_brands = st.multiselect('Выберите марки автомобилей', list(model_dict.keys()))
#
# selected_models = {brand: st.multiselect(f'Выберите модели для {brand}', model_dict[brand]) for brand in selected_brands}
#
# years = st.slider('Года', 2008, 2019, (2008, 2019))
#
# if st.button('Выполнить'):
#     main(cities, selected_brands, selected_models, years)
#
# if st.button('Очистить'):
#     st.cache_data.clear()
#     st.experimental_rerun()


# import io
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import seaborn as sns
#
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.linear_model import SGDRegressor
# from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
# from sklearn.metrics import mean_squared_error as mse, r2_score
# from sklearn.metrics import PredictionErrorDisplay
# import warnings
#
# warnings.filterwarnings('ignore')
# pd.options.display.max_columns = None
#
# translations = {
#     'year': 'год',
#     'distance(km)': 'пробег (км)',
#     'engine_capacity(L)': 'объем двигателя (л)',
#     'fuel_type_бензин': 'тип топлива - бензин',
#     'fuel_type_дизель': 'тип топлива - дизель',
#     'fuel_type_электро': 'тип топлива - электро',
#     'fuel_type_ГБО': 'тип топлива - ГБО',
#     'fuel_type_гибрид': 'тип топлива - гибрид'
# }
#
# class QuantileReplacer(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold=0.05):
#         self.threshold = threshold
#         self.quantiles = {}
#
#     def fit(self, X, y=None):
#         for col in X.select_dtypes(include='number'):
#             low_quantile = X[col].quantile(self.threshold)
#             high_quantile = X[col].quantile(1 - self.threshold)
#             self.quantiles[col] = (low_quantile, high_quantile)
#         return self
#
#     def transform(self, X):
#         X_copy = X.copy()
#         for col in X.select_dtypes(include='number'):
#             low_quantile, high_quantile = self.quantiles[col]
#             rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
#             if rare_mask.any():
#                 rare_values = X_copy.loc[rare_mask, col]
#                 replace_value = np.mean([low_quantile, high_quantile])
#                 if rare_values.mean() > replace_value:
#                     X_copy.loc[rare_mask, col] = high_quantile
#                 else:
#                     X_copy.loc[rare_mask, col] = low_quantile
#         return X_copy
#
# def main(cities, selected_brands, selected_models, years):
#     if not cities:
#         st.write("Пожалуйста, выберите город.")
#         return
#
#     if not selected_brands:
#         st.write("Пожалуйста, выберите марку автомобиля.")
#         return
#
#     if not any(selected_models.values()):
#         st.write("Пожалуйста, выберите модель автомобиля.")
#         return
#
#     df = pd.DataFrame()
#     if 'Екатеринбург' in cities:
#         if 'Hyundai' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_solaris.csv')
#         if 'Toyota' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_camry.csv')
#         if 'Lada' in selected_brands:
#             df = pd.read_csv('cars_data_ekb_lada.csv')
#     if 'Москва' in cities:
#         if 'Hyundai' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_solaris.csv')
#         if 'Toyota' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_camry.csv')
#         if 'Lada' in selected_brands:
#             df = pd.read_csv('cars_data_moscow_lada.csv')
#
#     if df.empty:
#         st.write("Нет данных для выбранных условий.")
#         return
#
#     df = df.dropna()
#
#     st.title(f"Анализ данных об автомобилях из города {' и '.join(cities)} и марки {', '.join(selected_brands)}")
#
#     st.subheader('Информация о данных')
#
#     num = ['year', 'distance(km)', 'engine_capacity(L)']
#     cat = ['fuel_type']
#
#     num_pipe = Pipeline([
#         ('QuantReplace', QuantileReplacer(threshold=0.01)),
#         ('norm', MinMaxScaler())
#     ])
#
#     cat_pipe = Pipeline([
#         ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
#     ])
#
#     preprocessors_All = ColumnTransformer(transformers=[
#         ('num', num_pipe, num),
#         ('cat', cat_pipe, cat)
#     ])
#
#     X, y = df.drop(columns=['price(rub)']), df['price(rub)']
#
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_train_prep = preprocessors_All.fit_transform(X_train)
#     X_val_prep = preprocessors_All.transform(X_val)
#
#     if X_train_prep.shape[1] != len(num) + len(cat):
#         st.write("Количество признаков после трансформации не совпадает с ожидаемым. Проверьте данные.")
#         return
#
#     model = SGDRegressor(random_state=42)
#     model.fit(X_train_prep, y_train)
#
#     def get_coefs(model):
#         B0 = model.intercept_[0]
#         B = model.coef_
#         return B0, B
#
#     def print_model(B0, B, features_names):
#         line = '{:.3f}'.format(B0)
#         sign = ['+', '-']
#         for p, (fn, b) in enumerate(zip(features_names, B)):
#             line = line + sign[int(0.5 * (np.sign(b) - 1))] + '{:.2f}*'.format(np.abs(b)) + fn
#         st.write('Уравнение линейной регрессии:')
#         st.write(f'Цена (руб) = {line}')
#         st.write('Где:')
#         for fn in features_names:
#             st.write(f'- {fn} = {translations.get(fn, fn)}')
#
#     def vis_weigths(weights, features_names=None, width=1200, height=600):
#         numbers = np.arange(0, len(weights))
#         if features_names:
#             tick_labels = np.hstack(['B0', features_names])
#         else:
#             tick_labels = ['B' + str(num) for num in numbers]
#         fig = go.Figure()
#         fig.add_trace(
#             go.Bar(x=numbers[weights < 0], y=weights[weights < 0], marker_color='red', name='Отрицательные веса'))
#         fig.add_trace(
#             go.Bar(x=numbers[weights >= 0], y=weights[weights >= 0], marker_color='blue', name='Положительные веса'))
#         fig.update_layout(
#             title="Веса модели",
#             width=width,
#             height=height,
#             template="plotly_dark",
#             xaxis=dict(tickmode='array', tickvals=numbers, ticktext=tick_labels)
#         )
#         st.plotly_chart(fig)
#
#     cat_features = preprocessors_All.transformers_[1][1]['encoder'].get_feature_names_out(cat)
#     features_names = list(num) + list(cat_features)
#
#     B0, B = get_coefs(model)
#     print_model(B0, B, features_names)
#
#     Bs = np.hstack([B0, B])
#     vis_weigths(Bs, features_names)
#
#     def calculate_metric(model_pipe, X, y, metric=r2_score, **kwargs):
#         y_model = model_pipe.predict(X)
#         return metric(y, y_model, **kwargs)
#
#     st.write(f"R2 на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train):.4f}")
#     st.write(f"R2 на валидационной выборке: {calculate_metric(model, X_val_prep, y_val):.4f}")
#
#     st.write(f"MSE на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train, mse):.4f}")
#     st.write(f"MSE на валидационной выборке: {calculate_metric(model, X_val_prep, y_val, mse):.4f}")
#
#     st.write(f"RMSE на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train, mse, squared=False):.4f}")
#     st.write(f"RMSE на валидационной выборке: {calculate_metric(model, X_val_prep, y_val, mse, squared=False):.4f}")
#
#     def cross_validation(X, y, model, scoring, cv_rule):
#         scores = cross_validate(model, X, y, scoring=scoring, cv=cv_rule)
#         st.write('Ошибка на кросс-валидации')
#         DF_score = pd.DataFrame(scores)
#         DF_score = DF_score.abs()  # Приводим к положительным значениям для удобства чтения
#         DF_score.columns = DF_score.columns.str.replace('-', '')
#         st.write(DF_score)
#         st.write('\n')
#         st.write(DF_score.mean()[2:])
#
#     scoring_reg = {'R2': 'r2', '-MSE': 'neg_mean_squared_error', '-MAE': 'neg_mean_absolute_error', '-Max': 'max_error'}
#     cross_validation(X_train_prep, y_train, model, scoring_reg, ShuffleSplit(n_splits=5, random_state=42))
#
#     fig, ax = plt.subplots()
#     PredictionErrorDisplay.from_predictions(
#         y_val, model.predict(X_val_prep), kind="actual_vs_predicted",
#         scatter_kwargs={"alpha": 0.5}, ax=ax
#     )
#     ax.set_xlabel("Предсказанные значения (руб)")
#     ax.set_ylabel("Фактические значения (руб)")
#     st.pyplot(fig)
#
#     def plot_regression_results(model, X, y, feature_names):
#         fig, axes = plt.subplots(1, len(feature_names), figsize=(15, 5))
#         for i, ax in enumerate(axes):
#             sns.regplot(x=X[:, i], y=y, ax=ax, line_kws={"color": "red"})
#             ax.set_xlabel(translations.get(feature_names[i], feature_names[i]))
#             ax.set_ylabel("Цена (руб)")
#         st.pyplot(fig)
#
#     plot_regression_results(model, X_val_prep, y_val, features_names)
#
#     def predict_next_year(model, X, features_names):
#         X_copy = X.copy()
#         if 'year' in features_names:
#             year_index = features_names.index('year')
#             X_copy[:, year_index] += 1
#             y_pred = model.predict(X_copy)
#             return y_pred
#         else:
#             st.write("Признак 'year' отсутствует в данных.")
#             return None
#
#     next_year_predictions = predict_next_year(model, X_val_prep, features_names)
#     if next_year_predictions is not None:
#         max_price = next_year_predictions.max()
#         mean_price = next_year_predictions.mean()
#         min_price = next_year_predictions.min()
#
#         st.write("### Прогнозируемая стоимость автомобилей на следующий год:")
#         st.markdown(f"<h2 style='color: blue;'>Максимальная: {max_price:,.2f} руб.</h2>", unsafe_allow_html=True)
#         st.markdown(f"<h2 style='color: blue;'>Средняя: {mean_price:,.2f} руб.</h2>", unsafe_allow_html=True)
#         st.markdown(f"<h2 style='color: blue;'>Минимальная: {min_price:,.2f} руб.</h2>", unsafe_allow_html=True)
#
# st.title('Анализ цен на автомобили')
# st.write('### Линейная регрессия цен на автомобили по годам')
#
# cities = st.multiselect('Выберите города', ['Екатеринбург', 'Москва'])
#
# model_dict = {
#     'Hyundai': ['Solaris'],
#     'Toyota': ['Camry'],
#     'Lada': ['Granta']
# }
#
# selected_brands = st.multiselect('Выберите марки автомобилей', list(model_dict.keys()))
#
# selected_models = {brand: st.multiselect(f'Выберите модели для {brand}', model_dict[brand]) for brand in selected_brands}
#
# years = st.slider('Года', 2008, 2019, (2008, 2019))
#
# if st.button('Выполнить'):
#     main(cities, selected_brands, selected_models, years)
#
# if st.button('Очистить'):
#     st.cache_data.clear()
#     st.experimental_rerun()


import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.metrics import PredictionErrorDisplay
import warnings

import logit_reg

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

translations = {
    'year': 'год',
    'distance(km)': 'пробег (км)',
    'engine_capacity(L)': 'объем двигателя (л)',
    'fuel_type_бензин': 'тип топлива - бензин',
    'fuel_type_дизель': 'тип топлива - дизель',
    'fuel_type_электро': 'тип топлива - электро',
    'fuel_type_ГБО': 'тип топлива - ГБО',
    'fuel_type_гибрид': 'тип топлива - гибрид'
}

class QuantileReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.quantiles = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include='number'):
            low_quantile = X[col].quantile(self.threshold)
            high_quantile = X[col].quantile(1 - self.threshold)
            self.quantiles[col] = (low_quantile, high_quantile)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X.select_dtypes(include='number'):
            low_quantile, high_quantile = self.quantiles[col]
            rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
            if rare_mask.any():
                rare_values = X_copy.loc[rare_mask, col]
                replace_value = np.mean([low_quantile, high_quantile])
                if rare_values.mean() > replace_value:
                    X_copy.loc[rare_mask, col] = high_quantile
                else:
                    X_copy.loc[rare_mask, col] = low_quantile
        return X_copy

def main(cities, selected_brands, selected_models, years):
    if not cities:
        st.write("Пожалуйста, выберите город.")
        return

    if not selected_brands:
        st.write("Пожалуйста, выберите марку автомобиля.")
        return

    if not any(selected_models.values()):
        st.write("Пожалуйста, выберите модель автомобиля.")
        return

    df = pd.DataFrame()
    if 'Екатеринбург' in cities:
        if 'Hyundai' in selected_brands:
            df = pd.read_csv('cars_data_ekb_solaris.csv')
        if 'Toyota' in selected_brands:
            df = pd.read_csv('cars_data_ekb_camry.csv')
        if 'Lada' in selected_brands:
            df = pd.read_csv('cars_data_ekb_lada.csv')
    if 'Москва' in cities:
        if 'Hyundai' in selected_brands:
            df = pd.read_csv('cars_data_moscow_solaris.csv')
        if 'Toyota' in selected_brands:
            df = pd.read_csv('cars_data_moscow_camry.csv')
        if 'Lada' in selected_brands:
            df = pd.read_csv('cars_data_moscow_lada.csv')

    if df.empty:
        st.write("Нет данных для выбранных условий.")
        return

    df = df.dropna()

    st.title(f"Анализ данных об автомобилях из города {' и '.join(cities)} и марки {', '.join(selected_brands)}")

    st.subheader('Информация о данных')

    num = ['year', 'distance(km)', 'engine_capacity(L)']
    cat = ['fuel_type']

    num_pipe = Pipeline([
        ('QuantReplace', QuantileReplacer(threshold=0.01)),
        ('norm', MinMaxScaler())
    ])

    cat_pipe = Pipeline([
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessors_All = ColumnTransformer(transformers=[
        ('num', num_pipe, num),
        ('cat', cat_pipe, cat)
    ])

    X, y = df.drop(columns=['price(rub)']), df['price(rub)']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_prep = preprocessors_All.fit_transform(X_train)
    X_val_prep = preprocessors_All.transform(X_val)

    if X_train_prep.shape[1] != len(num) + len(cat):
        st.write("Количество признаков после трансформации не совпадает с ожидаемым. Проверьте данные.")
        return

    model = SGDRegressor(random_state=42)
    model.fit(X_train_prep, y_train)

    def get_coefs(model):
        B0 = model.intercept_[0]
        B = model.coef_
        return B0, B

    def print_model(B0, B, features_names):
        line = '{:.3f}'.format(B0)
        sign = ['+', '-']
        for p, (fn, b) in enumerate(zip(features_names, B)):
            line = line + sign[int(0.5 * (np.sign(b) - 1))] + '{:.2f}*'.format(np.abs(b)) + fn
        st.write('Уравнение линейной регрессии:')
        st.write(f'Цена (руб) = {line}')
        st.write('Где:')
        for fn in features_names:
            st.write(f'- {fn} = {translations.get(fn, fn)}')

    def vis_weigths(weights, features_names=None, width=1200, height=600):
        numbers = np.arange(0, len(weights))
        if features_names:
            tick_labels = np.hstack(['B0', features_names])
        else:
            tick_labels = ['B' + str(num) for num in numbers]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=numbers[weights < 0], y=weights[weights < 0], marker_color='red', name='Отрицательные веса'))
        fig.add_trace(
            go.Bar(x=numbers[weights >= 0], y=weights[weights >= 0], marker_color='blue', name='Положительные веса'))
        fig.update_layout(
            title="Веса модели",
            width=width,
            height=height,
            template="plotly_dark",
            xaxis=dict(tickmode='array', tickvals=numbers, ticktext=tick_labels)
        )
        st.plotly_chart(fig)

    cat_features = preprocessors_All.transformers_[1][1]['encoder'].get_feature_names_out(cat)
    features_names = list(num) + list(cat_features)

    B0, B = get_coefs(model)
    print_model(B0, B, features_names)

    Bs = np.hstack([B0, B])
    vis_weigths(Bs, features_names)

    def calculate_metric(model_pipe, X, y, metric=r2_score, **kwargs):
        y_model = model_pipe.predict(X)
        return metric(y, y_model, **kwargs)

    st.write(f"R2 на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train):.4f}")
    st.write(f"R2 на валидационной выборке: {calculate_metric(model, X_val_prep, y_val):.4f}")

    st.write(f"MSE на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train, mse):.4f}")
    st.write(f"MSE на валидационной выборке: {calculate_metric(model, X_val_prep, y_val, mse):.4f}")

    st.write(f"RMSE на тренировочной выборке: {calculate_metric(model, X_train_prep, y_train, mse, squared=False):.4f}")
    st.write(f"RMSE на валидационной выборке: {calculate_metric(model, X_val_prep, y_val, mse, squared=False):.4f}")

    def cross_validation(X, y, model, scoring, cv_rule):
        scores = cross_validate(model, X, y, scoring=scoring, cv=cv_rule)
        st.write('Ошибка на кросс-валидации')
        DF_score = pd.DataFrame(scores)
        DF_score = DF_score.abs()  # Приводим к положительным значениям для удобства чтения
        DF_score.columns = DF_score.columns.str.replace('-', '')
        st.write(DF_score)
        st.write('\n')
        st.write(DF_score.mean()[2:])

    scoring_reg = {'R2': 'r2', '-MSE': 'neg_mean_squared_error', '-MAE': 'neg_mean_absolute_error', '-Max': 'max_error'}
    cross_validation(X_train_prep, y_train, model, scoring_reg, ShuffleSplit(n_splits=5, random_state=42))

    fig, ax = plt.subplots()
    PredictionErrorDisplay.from_predictions(
        y_val, model.predict(X_val_prep), kind="actual_vs_predicted",
        scatter_kwargs={"alpha": 0.5}, ax=ax
    )
    ax.set_xlabel("Предсказанные значения (руб)")
    ax.set_ylabel("Фактические значения (руб)")
    st.pyplot(fig)

    def plot_regression_results(model, X, y, feature_names):
        fig, axes = plt.subplots(1, len(feature_names), figsize=(15, 5))
        for i, ax in enumerate(axes):
            sns.regplot(x=X[:, i], y=y, ax=ax, line_kws={"color": "red"})
            ax.set_xlabel(translations.get(feature_names[i], feature_names[i]))
            ax.set_ylabel("Цена (руб)")
        st.pyplot(fig)

    plot_regression_results(model, X_val_prep, y_val, features_names)

    def predict_next_year(model, X, features_names):
        X_copy = X.copy()
        if 'year' in features_names:
            year_index = features_names.index('year')
            X_copy[:, year_index] += 1
            y_pred = model.predict(X_copy)
            return y_pred
        else:
            st.write("Признак 'year' отсутствует в данных.")
            return None

    next_year_predictions = predict_next_year(model, X_val_prep, features_names)
    if next_year_predictions is not None:
        max_price = next_year_predictions.max()
        mean_price = next_year_predictions.mean()
        min_price = next_year_predictions.min()

        st.write("### Прогнозируемая стоимость автомобилей на следующий год:")
        st.markdown(f"<h2 style='color: blue;'>Максимальная: {max_price:,.2f} руб.</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: blue;'>Средняя: {mean_price:,.2f} руб.</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: blue;'>Минимальная: {min_price:,.2f} руб.</h2>", unsafe_allow_html=True)

        # Построение графика с трендом
        current_years = X_val['year']
        next_year = current_years + 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=current_years, y=y_val, mode='markers', name='Текущие цены'))
        fig.add_trace(go.Scatter(x=next_year, y=next_year_predictions, mode='markers', name='Прогнозируемые цены', marker=dict(color='red')))
        fig.update_layout(title='Прогнозируемые цены на следующий год', xaxis_title='Год', yaxis_title='Цена (руб)')
        st.plotly_chart(fig)

st.title('Анализ цен на автомобили')
st.write('### Линейная регрессия цен на автомобили по годам')

cities = st.multiselect('Выберите города', ['Екатеринбург', 'Москва'])

model_dict = {
    'Hyundai': ['Solaris'],
    'Toyota': ['Camry'],
    'Lada': ['Granta']
}

selected_brands = st.multiselect('Выберите марки автомобилей', list(model_dict.keys()))

selected_models = {brand: st.multiselect(f'Выберите модели для {brand}', model_dict[brand]) for brand in selected_brands}

years = st.slider('Года', 2008, 2019, (2008, 2019))

if st.button('Выполнить'):
    main(cities, selected_brands, selected_models, years)

if st.button('Очистить'):
    st.cache_data.clear()
    st.experimental_rerun()

if st.button('Логистическая регрессия'):
    logit_reg.main()
