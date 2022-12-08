"""
2022.9.25
author:chy
使用LSTM_VMD进行预测，共7个特征，步长为60
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Flatten


data_dim = 8
epochs = 30
batch_size = 32
VMD_num = 5

df = pd.read_csv("./data3/VMDban-1.csv")

df_for_training = df[:-9685]
df_for_testing = df[-9685:]
# print(df_for_training.shape)
# print(df_for_testing.shape)

'''
可以注意到数据范围非常大，并且它们没有在相同的范围内缩放，
因此为了避免预测错误，让我们先使用MinMaxScaler缩放数据。
(也可以使用StandardScaler)
'''
scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)


# print(df_for_training_scaled)

def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


trainX, trainY = createXY(df_for_training_scaled, 60)
testX, testY = createXY(df_for_testing_scaled, 60)


def build_model():
    model = Sequential()

    model.add(LSTM(200, return_sequences=True, input_shape=trainX.shape[1:]))
    model.add(LSTM(100, return_sequences=True))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # grid_model.add(LSTM(50))
    # grid_model.add(Dropout(0.2))
    # grid_model.add(Dense(1))
    model.compile(loss='mse', metrics=['mape'], optimizer='Adam')
    return model
model = build_model()

model.fit(trainX,trainY,verbose=1,validation_data=(testX,testY),epochs=epochs,batch_size=batch_size)
prediction = model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-", prediction.shape)
'''
因为在缩放数据时，我们每行有 data_dim 列，现在我们只有 1 列是目标列。
所以我们必须改变形状来使用 inverse_transform
'''
prediction_copies_array = np.repeat(prediction, data_dim, axis=-1)
'''
data_dim列值是相似的，它只是将单个预测列复制了 4 次。所以现在我们有 5 列相同的值 。
'''
print(prediction_copies_array.shape)
"""
这样就可以使用 inverse_transform 函数。
"""
pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), data_dim)))[:, 0]

"""
但是逆变换后的第一列是我们需要的，所以我们在最后使用了 → [:,0]。
现在将这个 pred 值与 testY 进行比较，但是 testY 也是按比例缩放的，也需要使用与上述相同的代码进行逆变换。
"""
original_copies_array = np.repeat(testY, data_dim, axis=-1)
original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), data_dim)))[:, 0]

print("Pred Values-- ", pred)
print("\nOriginal Values-- ", original)

plt.plot(original, color='red', label='Real Wind Power(VMD)')
plt.plot(pred, color='blue', label='Predicted Wind Power(VMD)')
plt.title('Wind Power Prediction(VMD)')
plt.xlabel('Time')
plt.ylabel('Wind Power(VMD)')
plt.legend()
plt.show()

model.save('VMD{}_LSTM_epochs_{}_batch_size_{}.h5'.format(VMD_num, epochs, batch_size))
print('Model Saved!')
