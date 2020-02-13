# create test and training data sets

import pandas as pd 
import numpy as np 
import random
import matplotlib.pyplot as plt


# decide upon the time period of interest for generating buy signals
# Assume you can sell at the lows and buy at highs for that day only
# Assume some transaction cost say 50 bps
def long_returns(df, numdays):
	df['buyret'] = (df.Low / df.High.shift(numdays)-1)*100
	df.buyret.fillna(0, inplace=True)
	return df

# decide upon the time period of interest for generating sell signals
# Assume you can sell at the lows and buy at highs only
# Assume some transaction cost say 50 bps
def short_returns(df, numdays):
	df['sellret'] = (df.Low.shift(numdays) / df.High -1)*100
	df.sellret.fillna(0,inplace=True)
	return df

def label_data(df):
	df['Action'] = 'None'
	df.loc[df['buyret'] > 0.5, 'Action'] = 'Buy'
	df.loc[df['sellret'] > 0.5, 'Action'] = 'Sell'
#	df = df[df.columns.drop(['buyret','sellret'])]
	return df

def create_labeled_charts(sample_size):
	chartnum = 1
# select files where the number of rows is more than 500 since a 100 data points will be lost in computing moving averages
	i = 1
	while(i <= sample_size):
		test_num = random.sample(range(0,len(symlist)-1), 1)
		i = i + 1
		filename = 'csvdata/' + symlist.Symbol[test_num[0]] + '.csv'
		temp = pd.read_csv(filename)
		
		if(len(temp)<500):
			i = i -1
		else:

			temp = temp[['Date', 'symbolid', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']]
			fwdret = 30

			temp = long_returns(temp, fwdret)
			temp = short_returns(temp, fwdret)
			temp = label_data(temp)
			temp = temp[['Date', 'symbolid', 'AdjClose', 'Action']]

			rollmean = temp.AdjClose.rolling(window=20).mean()
			temp['mav20'] = rollmean
			rollmean = temp.AdjClose.rolling(window=50).mean()
			temp['mav50'] = rollmean
			rollmean = temp.AdjClose.rolling(window=100).mean()
			temp['mav100'] = rollmean

			# remove all rows where there is even one NaN
			temp = temp.dropna()

			# now create charts with 200 rows and label them as the label in the last row
			for j in range(200,len(temp),20):
				fig = plt.figure()
				plt.plot(temp.AdjClose[j-200:j])
				plt.plot(temp.mav20[j-200:j],color='orange')
				plt.plot(temp.mav50[j-200:j],color='magenta')
				plt.plot(temp.mav100[j-200:j],color='yellow')
				fig.savefig('charts/'+ temp.Action[j] + '_' + str(chartnum), dpi=fig.dpi)
				fig.clf()
				plt.close()
				chartnum = chartnum + 1
				
			print(str(i) + ' of ' + str(sample_size) + ' done')

# read the list of symbols file
symlist = pd.read_csv('csvdata/descdata.csv')

create_labeled_charts(200)
