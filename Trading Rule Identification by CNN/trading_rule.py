import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# in this trading rule if the 20 day moving average is over 50 day moving average is over 200 day moving average then buy in the reverse case sell
# the data is already scaled for 200 day moving average
def trading_rule_20_50_200(df):
	# initialise the new action column
	df['RuleAction'] = 'None'

	df.loc[((df['mav20'] > df['mav50']) & (df['mav50'] > 1)), 'RuleAction'] = 'Buy'
	df.loc[((df['mav20'] < df['mav50']) & (df['mav50'] < 1)), 'RuleAction'] = 'Sell'

	return df


csvfilename = 'train_50.csv'

data = pd.read_csv('sampledata/'+ csvfilename)

data = trading_rule_20_50_200(data)

ytest = np.array(pd.get_dummies(data.Action))
predict_valid = np.array(pd.get_dummies(data.RuleAction))

df = prediction_accuracy(ytest, predict_valid)

ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')

# now save the heatmap
ax.figure.savefig('figures/trading_rule_50_50.png')