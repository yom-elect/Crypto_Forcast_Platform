from flask import Flask, render_template, request, url_for, Markup, jsonify
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf

from feedparser import parse

from utils import crypto_symbols, create_dataset

app = Flask(__name__) #Initialize the flask App
 
 

@app.route('/')
@app.route('/first')
def first():
    feed = parse("https://www.coindesk.com/arc/outboundfeeds/rss/")

    data = []
    for val in feed['entries']:
        data.append({
            'title': val['title'],
            'date': val['published'],
            'author': val['author'],
            'summary': val['summary'],
            'image': val['media_content'][0]['url'],
            'cta': val['link']
        })
    return render_template('first.html', feed=data[:10])


@app.route('/prediction')
def prediction():
 	return render_template("prediction.html", coins=crypto_symbols)
    
@app.route('/predict',methods=['POST'])
def predict():
    int_feature = [x for x in request.form.values()]
    coin = int_feature[0]
    days = int(int_feature[1])

    from datetime import date

    today = date.today()

    # dd/mm/YY
    d1 = today.strftime("%Y-%m-%d")
    coin_data = yf.download([f'{coin}-USD'], start='2014-01-01', end=d1, interval='1d')
    coin_data = coin_data.dropna()
    coin_data = coin_data.reset_index()
    coin_data_close = coin_data['Close']
    close_price = np.array(coin_data_close).reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_price)

    ##splitting dataset into train and test split
    training_size = int(len(scaled_close) * 0.75)
    test_size = len(scaled_close) - training_size
    train_data, test_data = scaled_close[0:training_size, :], scaled_close[training_size:len(scaled_close), :1]
            # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    try:
        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        output = []
        n_steps = 100
        i = 0
        saved_model = load_model(f'crypto_models/{coin}')
        # scores = saved_model.evaluate(X_test, ytest)

        # LSTM_accuracy = (1 - scores) * 100
        while (i < days):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = saved_model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = saved_model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                i = i + 1
        return render_template('prediction.html', prediction_text=output[days-1], coins= crypto_symbols,)
    except IOError as e:
        # reshape input to be  [samples, time steps, features] which is required for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64, verbose=1)
        model.save(f'crypto_models/{coin}')

        # scores = model.evaluate(X_test, ytest)

        # LSTM_accuracy = (1 - scores) * 100

        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        output = []
        n_steps = 100
        i = 0
        saved_model = load_model(f'crypto_models/{coin}')
        while (i < days):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = saved_model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = saved_model.predict(x_input, verbose=0)
                print(scaler.inverse_transform(yhat))
                output.append(scaler.inverse_transform(yhat)[0])
                temp_input.extend(yhat[0].tolist())
                i = i + 1

        return render_template('prediction.html', prediction_text= output[days-1], coins= crypto_symbols)
 

if __name__=='__main__':
    app.run(debug=True)
