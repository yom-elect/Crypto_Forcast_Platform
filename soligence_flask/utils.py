import numpy as np

# define coin symbols
crypto_symbols = ['BTC', 'ETH', 'USDT', 'USDC', 'BNB',
'BUSD', 'XRP', 'ADA', 'SOL', 'DOGE',
'DAI', 'DOT', 'HEX', 'TRX',
'SHIB', 'LEO', 'WBTC', 'AVAX', 'YOUC',
'MATIC', 'STETH', 'UNI1', 'LTC', 'FTT',
'LINK', 'CRO', 'XLM', 'NEAR', 'ATOM', 'WTRX']


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)