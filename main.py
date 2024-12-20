import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
import requests
from matplotlib.animation import FuncAnimation
import sqlite3
import os
import sys
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import ParseMode
from aiogram.utils import executor
from dotenv import load_dotenv
load_dotenv()
# Генерация симулированных данных
def generate_data():
    np.random.seed(42)
    date_rng = pd.date_range(start='2023-01-01', end='2023-01-31', freq='T')
    data = {
        'timestamp': date_rng,
        'CPU_usage': np.random.normal(loc=50, scale=10, size=(len(date_rng))),
        'RAM_usage': np.random.normal(loc=60, scale=15, size=(len(date_rng))),
        'network_traffic': np.random.normal(loc=70, scale=20, size=(len(date_rng)))
    }
    df = pd.DataFrame(data)
    return df

# Предобработка данных
def preprocess_data(df):
    df.set_index('timestamp', inplace=True)
    df = df.resample('T').mean()  # Приведение к минутному интервалу
    df.fillna(method='ffill', inplace=True)  # Удаление пропусков
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled

# Визуализация данных в реальном времени
def plot_data_realtime(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title('Метрики нагрузки сервера с течением времени')
    ax.set_xlabel('Время')
    ax.set_ylabel('Нормализованная нагрузка')

    line1, = ax.plot([], [], label='Использование CPU')
    line2, = ax.plot([], [], label='Использование RAM')
    line3, = ax.plot([], [], label='Сетевой трафик')
    ax.legend()

    def init():
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(0, 1)
        return line1, line2, line3

    def update(frame):
        line1.set_data(df.index[:frame], df['CPU_usage'][:frame])
        line2.set_data(df.index[:frame], df['RAM_usage'][:frame])
        line3.set_data(df.index[:frame], df['network_traffic'][:frame])
        return line1, line2, line3

    ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, interval=100)
    plt.show()

    # Сохранение графика как изображения
    fig.savefig('server_load.png')

# ARIMA моделирование
def arima_forecast(df, steps=10):
    model = ARIMA(df['CPU_usage'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# LSTM моделирование
def lstm_forecast(df, steps=10):
    data = df['CPU_usage'].values
    data = data.reshape((-1, 1))
    generator = TimeseriesGenerator(data, data, length=10, batch_size=32)  # Увеличен размер батча
    
    model = Sequential()
    model.add(LSTM(20, activation='relu', input_shape=(10, 1)))  # Уменьшено количество нейронов
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(generator, epochs=20)  # Уменьшено количество эпох
    
    predictions = []
    current_batch = data[-10:].reshape((1, 10, 1))
    
    for _ in range(steps):
        pred = model.predict(current_batch)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    
    return predictions

# Оценка точности
def evaluate_model(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    print(f'Среднеквадратичная ошибка (MSE): {mse}')
    return mse

# Уведомления через Telegram API
def send_telegram_message(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    
    # Проверка ответа от API
    if response.status_code != 200:
        print(f"Ошибка при отправке сообщения: {response.status_code} - {response.text}")
    else:
        print("Сообщение успешно отправлено")
    
    return response.json()

# Отправка изображения в Telegram
def send_telegram_photo(photo_path, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        data = {"chat_id": chat_id}
        files = {"photo": photo}
        response = requests.post(url, data=data, files=files)
        
        # Проверка ответа от API
        if response.status_code != 200:
            print(f"Ошибка при отправке изображения: {response.status_code} - {response.text}")
        else:
            print("Изображение успешно отправлено")
        
        return response.json()

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS calculations
                 (timestamp TEXT, method TEXT, result TEXT)''')
    conn.commit()
    conn.close()

# Сохранение вычислений в базу данных
def save_to_db(method, result):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("INSERT INTO calculations (timestamp, method, result) VALUES (datetime('now'), ?, ?)", (method, str(result)))
    conn.commit()
    conn.close()

# Получение истории вычислений
def get_history():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("SELECT * FROM calculations ORDER BY timestamp DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return rows

# Обработка команды /restart
async def restart_process(message: types.Message):
    await message.reply("Перезапуск процесса...")
    os.execv(sys.executable, ['python'] + sys.argv)

# Обработка команды /history
async def send_history(message: types.Message):
    history = get_history()
    history_message = "Последние вычисления:\n"
    for record in history:
        history_message += f"{record[0]} - {record[1]}: {record[2]}\n"
    await message.reply(history_message)

# Обработка неизвестных команд
async def unknown_command(message: types.Message):
    await message.reply("Извините, я не понимаю эту команду.")

# Функция для определения тональности прогноза
def determine_tone(predictions, positive_threshold=0.7, negative_threshold=0.3):
    mean_prediction = np.mean(predictions)
    if mean_prediction > positive_threshold:
        return "Положительный"
    elif mean_prediction < negative_threshold:
        return "Отрицательный"
    else:
        return "Нейтральный"

# Функция для отображения и сохранения графика прогнозов LSTM
def plot_and_save_lstm_forecast(predictions, filename='lstm_forecast.png'):
    # Преобразование прогнозов в одномерный массив
    predictions_flat = [pred[0] for pred in predictions]
    
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_flat, marker='o', linestyle='-')
    plt.title('LSTM Прогноз')
    plt.xlabel('Шаги')
    plt.ylabel('Прогнозируемое значение')
    plt.grid(True)
    plt.savefig(filename)  # Сохранение графика в файл
    plt.close()  # Закрытие графика, чтобы избежать отображения

# Основной код
if __name__ == "__main__":
    init_db()
    telegram_token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    
    df = generate_data()
    df_preprocessed = preprocess_data(df)
    
    # Визуализация данных
    plot_data_realtime(df_preprocessed)

    # ARIMA прогноз
    arima_predictions = arima_forecast(df_preprocessed)
    arima_message = f"ARIMA Прогноз на следующие шаги: {arima_predictions}\n" \
                    f"Этот прогноз показывает ожидаемое использование CPU в ближайшие моменты времени. " \
                    f"Тональность прогноза: {determine_tone(arima_predictions)}"
    print(arima_message)
    send_telegram_message(arima_message, telegram_token, chat_id)

    # LSTM прогноз
    lstm_predictions = lstm_forecast(df_preprocessed)
    lstm_message = f"LSTM Прогноз на следующие шаги: {lstm_predictions}\n" \
                   f"Этот прогноз основан на модели LSTM и показывает ожидаемое использование CPU. " \
                   f"Тональность прогноза: {determine_tone(lstm_predictions)}"
    print(lstm_message)
    send_telegram_message(lstm_message, telegram_token, chat_id)

    # Отображение и сохранение графика LSTM прогноза
    plot_and_save_lstm_forecast(lstm_predictions)

    # Отправка графика LSTM прогноза в Telegram
    send_telegram_photo('lstm_forecast.png', telegram_token, chat_id)

    # Оценка точности
    mse = evaluate_model(df_preprocessed['CPU_usage'][-10:], arima_predictions)
    mse_message = f"Среднеквадратичная ошибка (MSE): {mse}"
    print(mse_message)
    send_telegram_message(mse_message, telegram_token, chat_id)

    # Сохранение результатов в базу данных
    save_to_db("ARIMA", arima_predictions)
    save_to_db("LSTM", lstm_predictions)

    # Уведомление о завершении
    completion_message = "Прогноз нагрузки завершен"
    send_telegram_message(completion_message, telegram_token, chat_id)

    # Отправка изображения
    send_telegram_photo('server_load.png', telegram_token, chat_id)

    # Настройка Telegram-бота с aiogram
    bot = Bot(token=telegram_token)
    dp = Dispatcher(bot)
    dp.middleware.setup(LoggingMiddleware())

    # Регистрация обработчиков команд
    dp.register_message_handler(restart_process, commands=['restart'])
    dp.register_message_handler(send_history, commands=['history'])
    dp.register_message_handler(unknown_command, content_types=types.ContentTypes.ANY)

    # Запуск бота
    executor.start_polling(dp, skip_updates=True)