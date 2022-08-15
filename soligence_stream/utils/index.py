from datetime import date
from math import gamma
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yfinance as yf

import xgboost as xgb
from sklearn.model_selection import train_test_split

crypto_symbols = ['BTC', 'ETH', 'LTC', 'BNB','USDT', 'USDC',
'BUSD', 'XRP', 'ADA', 'SOL', 'DOGE',
'DAI', 'DOT', 'WTRX', 'HEX', 'TRX',
'SHIB', 'LEO', 'WBTC', 'AVAX', 'YOUC',
'MATIC', 'STETH', 'UNI1', 'FTT',
'LINK', 'CRO', 'XLM', 'NEAR', 'ATOM']

@st.cache
def load_coin_prediction(coin, unit, projection):
    today = date.today()

    # dd/mm/YY
    d1 = today.strftime("%Y-%m-%d")
   
    coin_set = yf.download([f'{coin}-{unit}'], start="2013-09-17", end=d1)
    coin_set2 = coin_set.copy().dropna()

    predict = ['Open_Predict', 'High_Predict', 
    'Low_Predict', 'Volume_Predict', 'Close_Predict']

    coin_set2['Open_Predict']=coin_set2[['Open']].shift(-projection)
    coin_set2['High_Predict']=coin_set2[['High']].shift(-projection)
    coin_set2['Low_Predict']=coin_set2[['Low']].shift(-projection)
    coin_set2['Volume_Predict']=coin_set2[['Volume']].shift(-projection)
    coin_set2['Close_Predict']=coin_set2[['Close']].shift(-projection)

    coin_set3 = coin_set2.tail(projection)
    coin_set3 = np.array(coin_set3[['Open','High','Low','Close','Volume']])

    #Creating the dataset based on the Independent variable X as numpy array
    X = np.array(coin_set2[['Open','High','Low','Close','Volume']])
    X = X[:-projection]

    predic_result = {}

    for pred in predict:
        y = np.array(coin_set2[pred])
        y = y[:-projection]

        #Spliting the dataset into 80% training and 20% testing datasets
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', 
        gamma= 4.751686814941894, max_depth=5, min_child_weight=5.0,
        req_alpha=47.0, req_lambda=0.20312582098075171, colsample_bytree=0.7877187154112623)
        
        xg_reg.fit(X_train, y_train)
        y_pred = xg_reg.predict(X_test)

        model_accuracy = xg_reg.score(X_train, y_train)

        predic_result[pred] = xg_reg.predict(coin_set3)
    return predic_result, model_accuracy, y_pred, y_test
    

def get_ticker(coins):
    ticker = []
    for coin in coins:
        ticker.append(f'{coin}-USD')
    
    return ticker

@st.cache
def coins_relationship(coins):
    ticker = get_ticker(coins)
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    dataset = yf.download(ticker, start="2013-09-17", end=d1)

    CryptoClose = dataset.loc[:,"Close"].copy().dropna()
    norm = CryptoClose.div(CryptoClose.iloc[0]).mul(100)
    ret = CryptoClose.pct_change().dropna()

    return CryptoClose, ret, norm