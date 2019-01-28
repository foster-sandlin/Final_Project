#!/home/fostersandlin/anaconda3/bin/python3.6
import pandas as pd
import time
import sys
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import cgi
import cgitb
import zipfile
cgitb.enable()
form = cgi.FieldStorage()


ask = str(sys.argv[1])


coinDict = {"btc": "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=", "eth":
            "https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end=", "neo":
                "https://coinmarketcap.com/currencies/neo/historical-data/?start=20130428&end="}

ask = form.getselectedvalue('coin')

#input("Enter coin for data retrieval:")
input1 = coinDict.get(ask)



# get market info for bitcoin from the start of 2016 to the current day
desired_market_info = pd.read_html(coinDict.get(ask) + time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
desired_market_info = desired_market_info.assign(Date=pd.to_datetime(desired_market_info['Date']))
# when Volume is equal to '-' convert it to 0
#desired_market_info.loc[desired_market_info['Volume'] == "-", 'Volume'] = 0
# convert to int
desired_market_info['Volume'] = desired_market_info['Volume'].astype('int64')
# look at the first few rows
#print(desired_market_info.head())

# BITCOIN DATA

# get market info for coin from the start of 2016 to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=" + time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
# when Volume is equal to '-' convert it to 0
bitcoin_market_info.loc[bitcoin_market_info['Volume'] == "-", 'Volume'] = 0
# convert to int
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
# look at the first few rows
#print(bitcoin_market_info.head())

# COLUMN IDENTIFY
bitcoin_market_info.columns = [bitcoin_market_info.columns[0]] + ['bt_' + i for i in bitcoin_market_info.columns[1:]]
desired_market_info.columns = [desired_market_info.columns[0]] + ['cl_' + i for i in desired_market_info.columns[1:]]

# BITCOIN MARKET INFO GRAPH
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
ax1.set_ylabel('Closing Price ($)', fontsize=12)
ax2.set_ylabel('Volume ($ bn)',fontsize=12)
ax2.set_yticks([int('%d000000000' % i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_title("Bitcoin Market Info")
ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013,2019) for j in [1, 7]])
ax1.plot(bitcoin_market_info['Date'].astype(datetime.datetime), bitcoin_market_info['bt_Open*'])
ax2.bar(bitcoin_market_info['Date'].astype(datetime.datetime).values, bitcoin_market_info['bt_Volume'].values)
fig.tight_layout()
# fig.figimage(bitcoin_im, 100, 120, zorder=3,alpha=.5)
plt.savefig("C:/wamp64/www/crypto-predict_website/images/bitcoin_market.png")
#plt.show()



# DESIRED MARKET INFO GRAPH
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
ax1.set_yscale('log')
ax1.set_ylabel('Closing Price ($)', fontsize=12)
ax2.set_ylabel('Volume ($ bn)', fontsize=12)
ax2.set_yticks([int('%d000000000' % i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_title("Desired Market Info")
ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013, 2019) for j in [1, 7]])
ax1.plot(desired_market_info['Date'].astype(datetime.datetime), desired_market_info['cl_Open*'])
ax2.bar(desired_market_info['Date'].astype(datetime.datetime).values, desired_market_info['cl_Volume'].values)
fig.tight_layout()
# fig.figimage(cl_im, 300, 180, zorder=3, alpha=.6)
plt.savefig("C:/wamp64/www/crypto-predict_website/images/desired_market.png")
#plt.show()


# MERGED HEAD DATA
market_info = pd.merge(bitcoin_market_info, desired_market_info, on=['Date'])
market_info = market_info[market_info['Date'] >= '2016-01-01']
for coins in ['bt_', 'cl_']:
    kwargs = {coins+'day_diff': lambda x: (x[coins+'Close**']-x[coins+'Open*'])/x[coins+'Open*']}
    market_info = market_info.assign(**kwargs)
print(market_info.head())

# TRAIN TEST PLOT
split_date = '2017-06-01'
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013, 2019) for j in [1, 7]])
ax1.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] < split_date]['bt_Close**'],
         color='#B08FC7', label='Training')
ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['bt_Close**'],
         color='#8FBAC8', label='Test')
ax2.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] < split_date]['cl_Close**'],
         color='#B08FC7')
ax2.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['cl_Close**'], color='#8FBAC8')
ax1.set_xticklabels('')
ax1.set_ylabel('Bitcoin Price ($)', fontsize=12)
ax2.set_ylabel('Desired Coin Price ($)', fontsize=12)
plt.tight_layout()
ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.savefig("C:/wamp64/www/crypto-predict_website/images/train_test.png")
#plt.show()

 #trivial lag model: P_t = P_(t-1)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %d %Y') for i in range(2013, 2019) for j in [1, 7]])
ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['bt_Close**'].values, label='Actual')
ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= datetime.datetime.strptime(split_date, '%Y-%m-%d') -
         datetime.timedelta(days=1)]['bt_Close**'][1:].values, label='Predicted')

ax1.set_ylabel('Bitcoin Price ($)', fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.set_title('Simple Lag Model (Test Set)')

ax2.set_ylabel('Desired Coin Price ($)', fontsize=12)
ax2.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['cl_Close**'].values, label='Actual')
ax2.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= datetime.datetime.strptime(split_date, '%Y-%m-%d') -
         datetime.timedelta(days=1)]['cl_Close**'][1:].values, label='Predicted')

fig.tight_layout()
plt.savefig("C:/wamp64/www/crypto-predict_website/images/trivial_lag.png")
#plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(market_info[market_info['Date'] < split_date]['bt_day_diff'].values, bins=100)
ax2.hist(market_info[market_info['Date'] < split_date]['cl_day_diff'].values, bins=100)
ax1.set_title('Bitcoin Daily Price Changes')
ax2.set_title('Desired Coin Daily Price Changes')
plt.savefig("C:/wamp64/www/crypto-predict_website/images/daily_price_change.png")
#plt.show()

for coins in ['bt_', 'cl_']:
    kwargs = {coins+'close_off_high': lambda x: 2*(x[coins+'High']-x[coins+'Close**'])/(x[coins+'High']-x[coins+'Low'])-1,
              coins+'volatility': lambda x: (x[coins+'High']-x[coins+'Low'])/(x[coins+'Open*'])}
    market_info = market_info.assign(**kwargs)

model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'cl_']
                                   for metric in ['Close**', 'Volume', 'close_off_high', 'volatility']]]
#need to reverse the data frame so that subsequent rows represent later timepoints
model_data = model_data.sort_values(by='Date')
print(model_data.head())

training_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)
window_len = 10
norm_cols = [coin+metric for coin in ['bt_', 'cl_'] for metric in ['Close**', 'Volume']]

LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0]-1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['cl_Close**'][window_len:].values/training_set['cl_Close**'][:-window_len].values)-1

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['cl_Close**'][window_len:].values/test_set['cl_Close**'][:-window_len].values)-1

LSTM_training_inputs[0]

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)



def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


np.random.seed(202)
# initialise model architecture
cl_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
# model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['cl_Close**'][window_len:].values/training_set['cl_Close**'][:-window_len].values)-1
# train model on data
# cl_history contains information on the training error per epoch
cl_history = cl_model.fit(LSTM_training_inputs, LSTM_training_outputs,
                            epochs=50, batch_size=1, verbose=2, shuffle=True)

fig, ax1 = plt.subplots(1,1)

ax1.plot(cl_history.epoch, cl_history.history['loss'])
ax1.set_title('Training Error')

if cl_model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)

else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.savefig("C:/wamp64/www/crypto-predict_website/images/mae.png")
#plt.show()


fig, ax1 = plt.subplots(1, 1)
ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]])
ax1.set_xticklabels([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]])
ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['cl_Close**'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(cl_model.predict(LSTM_training_inputs))+1) * training_set['cl_Close**'].values[:-window_len])[0],
         label='Predicted')
ax1.set_title('Training Set: Single Timepoint Prediction')
ax1.set_ylabel('Desired Coin Price ($)', fontsize=12)
ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.annotate('MAE: %.4f' % np.mean(np.abs((np.transpose(cl_model.predict(LSTM_training_inputs))+1)-\
            (training_set['cl_Close**'].values[window_len:])/(training_set['cl_Close**'].values[:-window_len]))),
             xy=(0.75, 0.9),  xycoords='axes fraction',
             xytext=(0.75, 0.9), textcoords='axes fraction')

axins = zoomed_inset_axes(ax1, 3.35, loc=10) # zoom-factor: 3.35, location: centre
axins.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]])
axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
           training_set['cl_Close**'][window_len:], label='Actual')
axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
           ((np.transpose(cl_model.predict(LSTM_training_inputs))+1) * training_set['cl_Close**'].values[:-window_len])[0],
           label='Predicted')
axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
axins.set_ylim([10,60])
axins.set_xticklabels('')
mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.savefig("C:/wamp64/www/crypto-predict_website/images/trainset_timepoint.png")
#plt.show()

#TEST THIS
#date2017 = [datetime.time(2017, i+1, 1) for i in range(12)]
#date2018 = [datetime.time(2018, i+1, 1) for i in range(12)]
#total_date = [datetime.combine(date2017, date2018)]

fig, ax1 = plt.subplots(1, 1)
ax1.set_xticks(([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]]))
ax1.set_xticklabels(([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]]))
ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
         test_set['cl_Close**'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(cl_model.predict(LSTM_test_inputs))+1) * test_set['cl_Close**'].values[:-window_len])[0],
         label='Predicted')
ax1.annotate('MAE: %.4f' % np.mean(np.abs((np.transpose(cl_model.predict(LSTM_test_inputs))+1) -\
             (test_set['cl_Close**'].values[window_len:])/(test_set['cl_Close**'].values[:-window_len]))),
             xy=(0.75, 0.9),  xycoords='axes fraction',
             xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.set_title('Test Set: Single Timepoint Prediction', fontsize=13)
ax1.set_ylabel('Desired Coin Price ($)', fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.savefig("C:/wamp64/www/crypto-predict_website/images/mae_predicted.png")
#plt.show()

#Predict for next 5 days

np.random.seed(202)

pred_range = 5

cl_model = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)

LSTM_training_outputs = []
for i in range(window_len, len(training_set['cl_Close**'])-pred_range):
    LSTM_training_outputs.append((training_set['cl_Close**'][i:i+pred_range].values/
                                  training_set['cl_Close**'].values[i-window_len])-1)
LSTM_training_outputs = np.array(LSTM_training_outputs)


eth_history = cl_model.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs,
                            epochs=50, batch_size=1, verbose=2, shuffle=True)



np.random.seed(202)


pred_range = 5
bt_model = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)
LSTM_training_outputs = []
for i in range(window_len, len(training_set['bt_Close**'])-pred_range):
    LSTM_training_outputs.append((training_set['bt_Close**'][i:i+pred_range].values/
                                  training_set['bt_Close**'].values[i-window_len])-1)
LSTM_training_outputs = np.array(LSTM_training_outputs)

bt_history = bt_model.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs,
                            epochs=50, batch_size=1, verbose=2, shuffle=True)
eth_pred_prices = ((cl_model.predict(LSTM_test_inputs)[:-pred_range][::pred_range]+1)*\
                   test_set['cl_Close**'].values[:-(window_len + pred_range)][::5].reshape(int(np.ceil((len(LSTM_test_inputs)-pred_range)/float(pred_range))),1))
bt_pred_prices = ((bt_model.predict(LSTM_test_inputs)[:-pred_range][::pred_range]+1)*\
                   test_set['bt_Close**'].values[:-(window_len + pred_range)][::5].reshape(int(np.ceil((len(LSTM_test_inputs)-pred_range)/float(pred_range))),1))

pred_colors = ["#FF69B4", "#5D6D7E", "#F4D03F","#A569BD","#45B39D"]
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_xticks(([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]]))
ax1.set_xticks(([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]]))
ax1.set_xticklabels(([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]]))
ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
         test_set['bt_Close**'][window_len:], label='Actual')
ax2.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
         test_set['cl_Close**'][window_len:], label='Actual')
for i, (cl_pred, bt_pred) in enumerate(zip(eth_pred_prices, bt_pred_prices)):

    if i<5:
        ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
                 bt_pred, color=pred_colors[i%5], label="Predicted")
    else:
        ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
                 bt_pred, color=pred_colors[i%5])
    ax2.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
             cl_pred, color=pred_colors[i%5])
ax1.set_title('Test Set: 5 Timepoint Predictions',fontsize=13)
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax1.set_xticklabels('')
ax2.set_ylabel('Desired Coin Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.13, 1), loc=2, borderaxespad=0., prop={'size': 12})
fig.tight_layout()
plt.savefig("C:/wamp64/www/crypto-predict_website/images/testset_timepoint.png")
#plt.show()

pictures_zip = zipfile.ZipFile('/var/www/html/crypto-predict_website/images//predictions.zip','w')
pictures_zip.write('/var/www/html/crypto-predict_website/images/bitcoin_market.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.write('/var/www/html/crypto-predict_website/images/daily_price_change.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.write('/var/www/html/crypto-predict_website/images/desired_market.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.write('/var/www/html/crypto-predict_website/images/mae.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.write('/var/www/html/crypto-predict_website/images/mae_predicted.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.write('/var/www/html/crypto-predict_website/images/testset_timepoint.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.write('/var/www/html/crypto-predict_website/images/trainset_timepoint.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.write('/var/www/html/crypto-predict_website/images/train_test.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.write('/var/www/html/crypto-predict_website/images/trivial_lag.png',compress_type=zipfile.ZIP_DEFLATED)
pictures_zip.close()
