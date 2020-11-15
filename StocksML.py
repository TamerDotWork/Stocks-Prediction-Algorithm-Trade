
import flask
import math 
import pandas as pd
import pandas_datareader as web
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

NSDQ = web.DataReader('^IXIC', data_source='yahoo', start='1973-01-01' ,end='2020-11-10')
#NSDQ

#plt.figure(figsize=(20.5,6.5))
#plt.plot(NSDQ['Close'] , label = 'NASDAQ' , linewidth=.5)
#plt.title('nsdq index')
#plt.xlabel('years')
#plt.ylabel('price')
#plt.legend(loc='upper left')
#plt.show

data = NSDQ.filter(['Close'])
dataset = data.values
training_data_len = math.ceil( len(dataset) * .8 )
training_data_len

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#scaled_data

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i,0])
  if i<=61:
    print(x_train)
    print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train ,(x_train.shape[0], x_train.shape[1],1))
x_train.shape

model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape = (x_train.shape[1] , 1)))
model.add(LSTM(50,return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train,y_train,batch_size=1,epochs=1)

test_data = scaled_data[training_data_len - 60:,:]
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean( predictions - y_test)**2 )
#rmse

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
fig = plt.figure(figsize =(16,8))
plt.title('Model')
plt.xlabel('Data' , fontsize=12)
plt.ylabel('close price' , fontsize=12)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])


tmpfile = BytesIO()
fig.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

html = 'Some html head' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + 'Some more html'

with open('index.html','w') as f:
    f.write(html)
#valid




app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

app.run()
