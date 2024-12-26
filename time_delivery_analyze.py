import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
import joblib

# Загрузка и подготовка данных
data = pd.read_csv('Food_Delivery_Times.csv')
data = data.drop(columns=['Order_ID', 'Courier_Experience_yrs'])

# Обработка пропусков
data['Weather'].fillna(data['Weather'].mode()[0], inplace=True)
data['Traffic_Level'].fillna(data['Traffic_Level'].mode()[0], inplace=True)
data['Time_of_Day'].fillna(data['Time_of_Day'].mode()[0], inplace=True)
data['Vehicle_Type'].fillna(data['Vehicle_Type'].mode()[0], inplace=True)
data.dropna(subset=['Delivery_Time_min'], inplace=True)

# Определение признаков и целевой переменной (включаем Preparation_Time_min)
X = data.drop(columns=['Delivery_Time_min'])
y = data['Delivery_Time_min']

# Преобразование категориальных переменных в числовые
categorical_features = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Создание пайплайна для модели линейной регрессии
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Обучение модели
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Сохранение модели и признаков
feature_names = X.columns.tolist()
joblib.dump((model, feature_names), 'delivery_time_predictor.pkl')

class DeliveryTimeApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Ввод данных пользователем
        self.distance_input = TextInput(hint_text='Расстояние (км)', multiline=False)
        self.preparation_time_input = TextInput(hint_text='Время подготовки (мин)', multiline=False)
        self.weather_input = TextInput(hint_text='Погода (Clear/Windy/Snowy/Foggy)', multiline=False)
        self.traffic_input = TextInput(hint_text='Уровень трафика (Low/Medium/High)', multiline=False)
        self.time_input = TextInput(hint_text='Время суток (Morning/Afternoon/Evening/Night)', multiline=False)
        self.vehicle_input = TextInput(hint_text='Тип транспорта (Bike/Scooter/Car)', multiline=False)

        # Создание горизонтального layout для кнопки и результата
        result_layout = BoxLayout(orientation='horizontal')
        self.result_label = Label(text='Предсказанное время доставки: ')
        predict_button = Button(text='Предсказать время доставки')
        predict_button.bind(on_press=self.predict_delivery_time)
        # Создание результата
        result_layout.add_widget(predict_button)
        result_layout.add_widget(self.result_label)

        layout.add_widget(self.distance_input)
        layout.add_widget(self.preparation_time_input)
        layout.add_widget(self.weather_input)
        layout.add_widget(self.traffic_input)
        layout.add_widget(self.time_input)
        layout.add_widget(self.vehicle_input)
        layout.add_widget(result_layout)  # Добавление горизонтального layout

        return layout

    def predict_delivery_time(self, instance):
        # Получение введенных данных
        distance = float(self.distance_input.text)
        preparation_time = float(self.preparation_time_input.text)
        weather = self.weather_input.text.strip()
        traffic_level = self.traffic_input.text.strip()
        time_of_day = self.time_input.text.strip()
        vehicle_type = self.vehicle_input.text.strip()

        # Создание DataFrame для предсказания
        input_data = pd.DataFrame({
            'Distance_km': [distance],
            'Preparation_Time_min': [preparation_time],
            'Weather': [weather],
            'Traffic_Level': [traffic_level],
            'Time_of_Day': [time_of_day],
            'Vehicle_Type': [vehicle_type]
        })

        # Предсказание времени доставки
        predicted_time = model.predict(input_data)[0]

        # Отображение результата
        self.result_label.text = f'Предсказанное время доставки: {predicted_time:.0f} минут'


if __name__ == '__main__':
    DeliveryTimeApp().run()
