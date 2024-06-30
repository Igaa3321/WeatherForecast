#zaimportowanie potrzebnych bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

#lista miejscowości, dla których będzie sprawdzana predykcja temperatury
csv_cities = ['Gdańsk', 'Dolina Pięciu Stawów', 'Kraków', 'Łazy', 'Jarocin']

#iterowanie po plikach dla poszczególnych miast
for city in csv_cities:
    print(f'Trenowanie dla: {city}')
    city_modified = unidecode.unidecode(city)
    csv_file = "new_data_" + city_modified.replace(" ", "") + ".csv"
    print(f'Plik: {csv_file}')
    data = pd.read_csv(csv_file, encoding='ISO-8859-2') #wczytanie danych

    #zmiana polskich nazw Rok, Miesiac itd. na angielskie -->funkcja datetime nie obsluguje innego jezyka
    data['Data'] = pd.to_datetime(data.rename(columns={'Rok': 'year', 'Miesiac': 'month', 'Dzien': 'day', 'Godzina': 'hour'})[['year', 'month', 'day', 'hour']])
    data.set_index('Data', inplace=True)

    #lista czynnikow meteorologicznym, ktore wzieto pod uwage do predykcji temperatury
    columns_of_interest = ['Temperatura powietrza', 'Predkosc wiatru [m/s]', 'Zachmurzenie ogolne']

    data = data[columns_of_interest].dropna() # przetwarzanie danych wejściowych przed użyciem ich do trenowania modelu
    # i usuniecie brakujacych wartosci

    #interpolacja danych i sprawdzenie barukacych wartosci
    data.interpolate(method='time', inplace=True)
    print(data.isna().sum())

    #normalizacja danych
    scalers = {}
    for col in columns_of_interest:
        scalers[col] = MinMaxScaler(feature_range=(0, 1))
        data[col] = scalers[col].fit_transform(data[[col]])


    #tworzenie sekwencji
    def create_sequences(data, seq_length):
        x = []
        y = []
        dates = []
        for i in range(len(data) - seq_length):
            x.append(data.iloc[i:i + seq_length].values)
            y.append(data.iloc[i + seq_length, 0])
            dates.append(data.index[i + seq_length])
        return np.array(x), np.array(y), dates


    seq_length = 30  # Długość sekwencji, którą będziemy używać do predykcji
    x, y, dates = create_sequences(data, seq_length)

    # Podział na dane treningowe i testowe
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    dates_train, dates_test = dates[:split], dates[split:]

    # Sprawdzenie kształtów powstałych tablic
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Budowanie modelu LSTM
    model = Sequential()
    model.add(Input(shape=(seq_length, len(columns_of_interest))))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Kompilacja modelu
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Trenowanie modelu
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Predykcje na zbiorze testowym
    y_pred = model.predict(x_test)

    # Odwrotna transformacja skali do oryginalnych wartości
    y_test_transformed = scalers['Temperatura powietrza'].inverse_transform(y_test.reshape(-1, 1))
    y_pred_transformed = scalers['Temperatura powietrza'].inverse_transform(y_pred)

    # Filtrowanie danych testowych tylko na rok 2023
    mask_2023 = np.array([date.year == 2023 for date in dates_test])
    dates_test_2023 = np.array(dates_test)[mask_2023]
    y_test_transformed_2023 = y_test_transformed[mask_2023]
    y_pred_transformed_2023 = y_pred_transformed[mask_2023]

    # Obliczanie błędów tylko na rok 2023
    mse_2023 = mean_squared_error(y_test_transformed_2023, y_pred_transformed_2023)
    print(f'MSE: {mse_2023}')
    print(f'RMSE: {math.sqrt(mse_2023)}')
    print(f'Średnia różnica między przewidywaną temperaturą a rzeczywistą temperaturą w 2023 roku w mieście {city} wynosi około {round(math.sqrt(mse_2023), 2)} stopni(a) Celsjusza.')

    # Wizualizacja wyników dla danych testowych
    plt.figure(figsize=(14, 5))
    plt.plot(dates_test_2023, y_test_transformed_2023, 'b.', linestyle='None', markersize=2, label='Prawdziwa temperatura')
    plt.plot(dates_test_2023, y_pred_transformed_2023, 'r.', linestyle='None', markersize=2, label='Prognozowana temperatura')
    plt.title(f'Predykcja temperatury dla miasta {city}')
    plt.xlabel('Czas')
    plt.ylabel('Temperatura [°C]')
    plt.legend()
    plt.show(block=False)

    print(f'Zakres danych treningowych: {dates_train[0]} - {dates_train[-1]}')
    print(f'Zakres danych testowych: {dates_test[0]} - {dates_test[-1]}')

plt.show()
