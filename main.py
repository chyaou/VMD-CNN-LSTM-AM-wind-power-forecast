import re

import numpy as np
from tensorflow.python.keras.layers import Dropout, Dense, SimpleRNN, LSTM
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def select():
    path = './models'
    a = os.listdir(path)
    c = []
    b = sorted(a)
    f = []
    for i, j in enumerate(b):
        c.append(float((re.findall(r"\d+\.?\d*", j[0:5]))[0]))
        f.append(j)
    # d = sorted(np.array(c), key=float)
    # e = argsort(d)
    # g = f[int(e[0])]
    # print(g)
    d = c.index(min(c))
    d = f[d]
    print(d)
    return d
pre_time = 5
mem_his_days = 10
unit = 100
dense_layers = 'lstm+rnn'
lstm_layers = 3
epochs = 1
test_size=0.1
data = pd.read_csv(r'./data2/handle_Turbine_Data .csv')
df = pd.DataFrame(data)
df.dropna(inplace=True)

df.sort_index(inplace=True)

# print(df.head())
df['label'] = df['ActivePower'].shift(-pre_time)
# # print(df['label'])
# # print(df.iloc[:,:-1])
scaler = StandardScaler()
sca_X = scaler.fit_transform(df.iloc[:,1:-1])
from collections import deque
deq = deque(maxlen=mem_his_days)
X = []
for i in sca_X:
    deq.append(list(i))
    if len(deq) == mem_his_days:
        X.append(list(deq))
X_lately = X[-pre_time:]
X = X[:-pre_time]
# print(len(X))
# print(len(X_lately))
y = df['label'].values[mem_his_days-1:-pre_time]
X = np.array(X)
y = np.array(y)
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
filepath = './models/{val_mape:.4f}_{epoch:02d}_'+f'men_{mem_his_days}_lstm_{lstm_layers}_dense_{dense_layers}_unit_{unit}_test_size_{test_size}'
checkpoint = ModelCheckpoint(
    filepath=filepath,
    save_weights_only=False,
    monitor='val_mape',
    mode='min',
    save_best_only=True
                    )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,shuffle=False)

y_drame = pd.DataFrame(y_test)
y_drame.to_excel("data.xlsx", sheet_name="data", index=False)
model = Sequential()
model.add(LSTM(unit, input_shape=X.shape[1:],activation='relu',return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(unit,activation='relu',return_sequences=True))
model.add(Dropout(0.1))

# model.add(SimpleRNN(unit,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(unit, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer='Adam',loss='mae',metrics=['mape'])
model.fit(X_train, y_train, batch_size=64,epochs=epochs,validation_data=(X_test, y_test),callbacks=[checkpoint])

from tensorflow.python.keras.models import load_model

best_model = load_model('./models/'+ str(select()))
pre = best_model.predict(X_test)
print(pre[0:-1])

import matplotlib.pyplot as plt
# df_time = df['date'][-len(y_test):].values
# print(df_time)
# pre = pre.reshape(724,)
# print(pre)
# print(y_test)

plt.plot(y_test,color='red',label='price')
plt.plot(pre[2],color='green',label='predict')
plt.savefig('./save.png')
plt.show()
