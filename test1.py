from ucimlrepo import fetch_ucirepo
import pandas

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
#print(adult.metadata)

# variable information
#print(adult.variables)

#1 Подсчет количества столбцов
full = pandas.concat([X, y], axis=1)
number_col = len(full.columns)
print("Количество столбцов: ", number_col)

#2 Есть ли пропуски в данных? Если есть, то в каких столбцах
# Проверка на пропуски в данных
miss = (X == '?').sum()

# Вывод столбцов с пропусками в X
print("Пропуски в столбцах:")
if miss.sum() > 0:
    print(miss[miss > 0])
else:
    print("Пропусков нет")

#3 Кол-во уникальных значений в столбце race
number_unique_race = X['race'].nunique()
print(f"Количество уникальных значений в столбце race: {number_unique_race}")

#4 Медиана hours-per-week
print(f"Медиана для hours-per-week: {X['hours-per-week'].median()}")

#5 Кого больше - женщин или мужчин с ЗП >50K?
dt = pandas.concat([X, y], axis=1)
dt50 = dt[dt['income'] == '>50K']
n_men = dt50[dt50['sex'] == 'Male'].shape[0]
n_women = dt50[dt50['sex'] == 'Female'].shape[0]
if n_men > n_women:
    print("Мужчин с доходом больше 50K больше.")
elif n_women > n_men:
    print("Женщин с доходом больше 50K больше.")
else:
    print("Количество мужчин и женщин с доходом больше 50K одинаково.")

#6 Заполните пропущенные данные в отдельных столбцах наиболее встречаемыми значениями.
for column in X.columns:
    most_frequent_value = X[column].mode()[0]
    X.loc[X[column] == '?', column] = most_frequent_value
print("После замены:")
print(X.isin(['?']).sum())