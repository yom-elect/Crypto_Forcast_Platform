import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from utils.index import crypto_symbols, load_coin_prediction, coins_relationship

st.set_page_config(layout="wide")

image = Image.open('logo.png')

st.image(image, width = 700)

st.title('SOLiGence App')
st.markdown("""
This is a leading financial multinational organisation that deals with 
stock and shares, saving and  **investments**!

""")

expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, requests, json, time
* **Data source:** [YahooFinance](https://finance.yahoo.com/cryptocurrencies/).
""")

col1 = st.sidebar
col2, col3 = st.columns((2,1))

col1.header('Users Input Options')

## Sidebar - Currency price unit
currency_price_unit = col1.selectbox('Select currency for price', ('USD', 'BTC', 'ETH'))

selected_coin = col1.selectbox('Select Coin to Predict', tuple(crypto_symbols))

prediction_range = col1.slider('Number of Days', 1, 30)

invest_amount = col1.text_input("Enter An Amount to Invest", 0)

col2.subheader(f'Price Prediction for {selected_coin}')
df, model_accuracy, pred_val, test_val = load_coin_prediction(selected_coin, currency_price_unit, prediction_range)
col2.dataframe(pd.DataFrame(df))

col3.subheader(f'Model Accuracy')
col3.write(f'{round(model_accuracy * 100, 2)}%')

plt.plot(figsize = (15,8) , fontsize = 13)
plt.plot(test_val, label='Original')
plt.plot(pred_val, label='Predicted')
plt.legend()
col3.pyplot(plt)

col3.subheader(f'R Squared Value')
R_squared = r2_score(test_val, pred_val)
col3.write(f'{round(R_squared * 100, 2)}%')


selected_coin = col1.multiselect('Coin Performance Relationship', crypto_symbols, crypto_symbols[:4])
CryptoClose, ret, norm = coins_relationship(selected_coin)

if float(invest_amount) > 0 and currency_price_unit == 'USD':
    if st.button('Predict Profit On Investment'):
        prices = (df['Close_Predict'])
        profit = 0
        amount = float(invest_amount)
        i = 0

        while i < len(prices) - 1:
            diff = (prices[i+1] - prices[i] ) / prices[i]
            profit += diff * amount
            i += 1

        status = 'loss' if (profit < 0) else 'profit'
        col2.write(f' You would have a {status} of {"..."}  {round(profit, 3)}USD after {prediction_range} days')

if st.button('Intercorrelation Close Prices'):
    CryptoClose.plot(figsize = (15,8) , fontsize = 13)
    plt.legend(fontsize = 15)
    col2.pyplot(plt)

if st.button('Coins Performamce'):
    norm.plot(figsize=(15,8), fontsize=13)
    plt.legend(fontsize=13)
    col2.pyplot(plt)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    plt.figure(figsize=(12,8))
    sns.set(font_scale=1.5)
    sns.heatmap(ret.corr(), cmap = "Reds", annot = True, annot_kws={"size":15}, vmax=1)
    col2.pyplot(plt)

if st.button('Return Visualization'):
    ret.plot(kind="hist", figsize = (12,8), bins=100)
    plt.legend(fontsize=13)
    col2.pyplot(plt)

if len(selected_coin) == 4:
    if st.button('Moving Average'):
        # compute a short-term 20-day moving average
        MA20 = CryptoClose.rolling(20).mean()
        # compute a Long-term 50-day moving average
        MA50 = CryptoClose.rolling(100).mean()
        # compute a Long-term 100-day moving average
        MA100 = CryptoClose.rolling(100).mean()
        fig, axs = plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.3, 'wspace': 0.1})
        axs[0,0].plot(CryptoClose[f'{selected_coin[0]}-USD'], label= 'price')
        axs[0,0].plot(MA20[f'{selected_coin[0]}-USD'], label= 'MA20')
        axs[0,0].plot(MA100[f'{selected_coin[0]}-USD'], label= 'MA100')
        axs[0,0].set_title(f'{selected_coin[0]}')
        axs[0,0].legend()
        axs[0,1].plot(CryptoClose[f'{selected_coin[1]}-USD'], label= 'price')
        axs[0,1].plot(MA20[f'{selected_coin[1]}-USD'], label= 'MA20')
        axs[0,1].plot(MA100[f'{selected_coin[1]}-USD'], label= 'MA100')
        axs[0,1].set_title(f'{selected_coin[1]}')
        axs[0,1].legend()
        axs[1,0].plot(CryptoClose[f'{selected_coin[2]}-USD'], label= 'price')
        axs[1,0].plot(MA20[f'{selected_coin[2]}-USD'], label= 'MA20')
        axs[1,0].plot(MA100[f'{selected_coin[2]}-USD'], label= 'MA100')
        axs[1,0].set_title(f'{selected_coin[2]}')
        axs[1,0].legend()
        axs[1,1].plot(CryptoClose[f'{selected_coin[3]}-USD'], label= 'price')
        axs[1,1].plot(MA20[f'{selected_coin[3]}-USD'], label= 'MA20')
        axs[1,1].plot(MA100[f'{selected_coin[3]}-USD'], label= 'MA100')
        axs[1,1].set_title(f'{selected_coin[3]}')
        axs[1,1].legend()
        col2.pyplot(plt)