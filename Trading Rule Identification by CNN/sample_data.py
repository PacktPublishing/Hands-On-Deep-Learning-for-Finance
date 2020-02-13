# create test and training data sets

import pandas as pd 
import numpy as np 
import random

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

# flexible function for computing moving average values
# normalise with variable that has the highest value
def moving_avg_data(df, mavnames, mavdays):
	if(len(mavnames) != len(mavdays)):
		print('Variable Names and Number of days must match')
		return
	
	for i in range(0,len(mavnames)):
		df[mavnames[i]] = df.AdjClose.rolling(window = mavdays[i]).mean()

	maxmovavg = mavnames[mavdays.index(max(mavdays))]
	mavnames.remove(maxmovavg)

	for i in range(0,len(mavnames)):
		df[mavnames[i]] = df[mavnames[i]] / df[maxmovavg]

	df.loc[:,maxmovavg] = 1
	df.drop(df.index[:max(mavdays)],inplace=True)
	return df

def create_datasets(csvfilename, sample_size):
	# choose random integers equal to sample_size to select stocks
	test_num = random.sample(range(0,len(symlist)-1), sample_size)

	# now read each file and label the data as buy or sell
	# create the moving average days and names list to create the dataframe
	# number of days forward return you would like to predict

	data = pd.DataFrame()

	for i in range(0,len(test_num)):
		filename = 'csvdata/' + symlist.Symbol[test_num[i]] + '.csv'
		temp = pd.read_csv(filename)
		temp = temp[['Date', 'symbolid', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']]

		mavnames = ['mav5', 'mav10','mav20','mav30','mav50','mav100','mav200']
		mavdays = [5,10,20,30,50,100,200]
		fwdret = 30

		temp = long_returns(temp, fwdret)
		temp = short_returns(temp, fwdret)
		temp = label_data(temp)
		temp = moving_avg_data(temp, mavnames, mavdays)
		temp = temp[['Date','symbolid','buyret','sellret','Action','mav5', 'mav10','mav20','mav30','mav50','mav100']]
		temp = temp.dropna()
		data = data.append(temp)

		#print(str(i/len(test_num)*100) + ' percent setup done')
	data.to_csv('sampledata/'+csvfilename)
	print(csvfilename + ' written to disk')

# read the list of symbols file
symlist = pd.read_csv('csvdata/descdata.csv')

create_datasets('train_50.csv', 50)

create_datasets('test_50.csv', 50)

create_datasets('train_250.csv', 250)

create_datasets('test_100.csv', 100)

create_datasets('train_500.csv', 500)

create_datasets('test_150.csv',100)

create_datasets('train_750.csv', 750)

create_datasets('test_250.csv',250)

create_datasets('train_1000.csv', 1000)

create_datasets('test_500.csv',500)

