import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn import preprocessing as prep
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression

# Настройки отображения
sns.set()
RANDOM_STATE = 42
pd.options.display.max_columns = None


# Функции для выполнения различных этапов анализа данных
def load_data():
    df = pd.read_csv("cars_drom_mashin.csv", delimiter=',')
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df['Year'] = 2024 - df['Year']
    df.rename(columns={'Year': 'Age', 'Transmission': 'Коробка передач'}, inplace=True)
    return df


def exploratory_data_analysis(df):
    cat_columns = [col for col in df.columns if df[col].dtypes == object]
    num_columns = [col for col in df.columns if df[col].dtypes != object]
    num_columns[0] = 'Age'
    st.write('Категориальные данные:', cat_columns, 'Число столбцов =', len(cat_columns))
    st.write('Числовые данные:', num_columns, 'Число столбцов =', len(num_columns))
    st.write(df.head())
    st.write(df[cat_columns].nunique())
    classification_df = df.copy()
    fig1, ax1 = plt.subplots()
    sns.histplot(df, x='Коробка передач', ax=ax1)
    ax1.set_title('Распределение коробок передач')
    st.pyplot(fig1)
    fig2 = sns.pairplot(
        data=classification_df[['Distance', 'Engine_capacity(cm3)', 'Price(euro)', 'Age', 'Коробка передач']],
        hue='Коробка передач', palette='rocket', height=3)
    st.pyplot(fig2)


def preprocess_data(df):
    data = df.drop(columns=['Коробка передач'])
    target = df['Коробка передач']
    cat_columns = [col for col in data.columns if data[col].dtypes == object]
    num_columns = [col for col in data.columns if data[col].dtypes != object]
    Label = prep.LabelEncoder()
    Label_encoded = Label.fit_transform(target)
    X_train, X_val, y_train, y_val = train_test_split(data, Label_encoded, test_size=0.2, random_state=RANDOM_STATE)

    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessors = ColumnTransformer(transformers=[
        ('num', numerical_pipe, num_columns),
        ('cat', categorical_pipe, cat_columns)
    ])

    preprocessors.fit(X_train)
    train_data = preprocessors.transform(X_train)
    val_data = preprocessors.transform(X_val)
    columns = np.append(num_columns, preprocessors.transformers_[1][1]['encoder'].get_feature_names_out(cat_columns))
    train_df = pd.DataFrame(train_data, columns=columns)
    val_df = pd.DataFrame(val_data, columns=columns)

    return X_train, X_val, y_train, y_val, train_df, val_df, preprocessors


def train_model(X_train, y_train, preprocessors):
    pipe = Pipeline([
        ('preprocessors', preprocessors),
        ('model', LogisticRegression(C=0.001, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe, X_train, y_train, X_val, y_val):
    def accuracy(model, X, y):
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)

    def recall(model, X, y):
        y_pred = model.predict(X)
        return recall_score(y, y_pred)

    train_acc = accuracy(pipe, X_train, y_train)
    val_acc = accuracy(pipe, X_val, y_val)
    train_recall = recall(pipe, X_train, y_train)
    val_recall = recall(pipe, X_val, y_val)

    return train_acc, val_acc, train_recall, val_recall


# Основная функция для выполнения всех шагов анализа данных
def main():
    df = load_data()
    st.write("### Исходные данные")
    st.write(df.head())

    st.write("### Исследовательский анализ данных")
    exploratory_data_analysis(df)

    st.write("### Предварительная обработка данных")
    X_train, X_val, y_train, y_val, train_df, val_df, preprocessors = preprocess_data(df)
    st.write("Данные для тренировки")
    st.write(train_df.head())
    st.write("Данные для валидации")
    st.write(val_df.head())

    st.write("### Обучение модели")
    pipe = train_model(X_train, y_train, preprocessors)
    train_acc, val_acc, train_recall, val_recall = evaluate_model(pipe, X_train, y_train, X_val, y_val)
    st.write(f"Точность на тренировочной выборке: {train_acc:.4f}")
    st.write(f"Точность на валидационной выборке: {val_acc:.4f}")
    st.write(f"Полнота на тренировочной выборке: {train_recall:.4f}")
    st.write(f"Полнота на валидационной выборке: {val_recall:.4f}")


