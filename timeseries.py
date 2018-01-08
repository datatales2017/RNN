# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:14:11 2017

@author: manasa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('all_stocks_1yr.csv')
df.head()
nulls = df.isnull().sum()
listings = df['Name'].unique()

def stocksmean(val1,val2):
    mean = df.groupby(['val1'])['val2'].mean()
    return mean
    
#finding the avgerage of Opening price
Open_avg = df.groupby(['Name'])['Open'].mean()
High_avg = df.groupby(['Name'])['High'].mean()
Low_avg = df.groupby(['Name'])['Low'].mean()
Close_avg = df.groupby(['Name'])['Low'].mean()

OpenNulls = []
for list in listings:
    OpenNulls.append(df[(df['Name']==list)&df['Open'].isnull()])
    
data = df.dropna()    
data.isnull().sum()

afl = data[data['Name']=='AFL']

afl.info()
afl.head()

afl['Date']
close = afl['Close']

ts = afl[['Date','Close']]
plt.plot(afl['Close'])
plt.plot(afl['Open'])
plt.plot(afl['High'])
plt.plot(afl['Low'])

ts.head()
plt.plot(ts)

cl = data[data['Name']=='AFL'].Close
#scaling the data
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
cl1 = cl.reshape(cl.shape[0],1)
cl = scl.fit_transform(cl1)
plt.plot(cl)


def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)
    
X,y = processData(cl,7)    

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=13)

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(LSTM(units=256,input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
history = model.fit(X_train,y_train,epochs=300,validation_data=(X_test,y_test),shuffle=False)

Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
plt.plot(scl.inverse_transform(Xt))

### TIME SERIES ANALYSIS OF ANOTHER STOCK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM

df = pd.read_csv('AOS_data.csv')
df.head(10)
nulls = df.isnull().sum()
plt.plot(df['Close'])
plt.plot(df['Volume'])

data = df['Close']

data[0:7,0]

np.shape(data)[0]
len(data)

from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
data = data.reshape(np.shape(data)[0],1)
data = scl.fit_transform(data)
plt.plot(data)

def processdata(data,cl):
    X,y = [],[]
    for i in range(len(data)-cl-1):
        X.append(data[i:i+cl,0])
        y.append(data[i+cl,0])
    return np.array(X),np.array(y)
    
X,y = processdata(data,5)

#SPLITING THE DATA
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=13)

model = Sequential()
model.add(LSTM(units=256,input_shape=(5,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
model.fit(X_train,y_train,epochs=200,validation_data=(X_test,y_test))

xpred = model.predict(X_test)
plt.plot(scl.inverse_transform(xpred))
plt.plot(y_test)