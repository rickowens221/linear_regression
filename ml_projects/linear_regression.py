import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.datasets import fetch_california_housing
fetch_california_housing = fetch_california_housing()
data = pd.DataFrame(fetch_california_housing.data, columns=fetch_california_housing.feature_names)
data['PRICE'] = fetch_california_housing.target

X, y = data.drop('PRICE', axis=1), data['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse}")

# Создание диаграммы рассеяния
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Предсказанные цены')

# Добавление диагональной линии y = x
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Идеальная модель')

# Добавление подписей осей и заголовка
plt.xlabel("Фактические цены")
plt.ylabel("Предсказанные цены")
plt.title("Фактические vs Предсказанные цены домов")

# Добавление сетки
plt.grid(True)

# Добавление легенды
plt.legend()

# Отображение графика
plt.show()
