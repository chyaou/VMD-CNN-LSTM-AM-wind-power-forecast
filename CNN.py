"""
2022.9.18 21:50
author:chy
使用CNN进行预测，共14个特征，步长为60

"""
import math

import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten


data_dim = 8
epochs = 30
batch_size = 32
df = pd.read_csv("./data2/handle_Turbine_Data .csv")

df_for_training = df[:-9685]
df_for_testing = df[-9685:]


'''
可以注意到数据范围非常大，并且它们没有在相同的范围内缩放，
因此为了避免预测错误，让我们先使用MinMaxScaler缩放数据。
(也可以使用StandardScaler)
'''
scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)




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
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mse', metrics=['mape'], optimizer='Adam')
    return model


model = build_model()
model.fit(trainX,trainY,epochs=epochs, verbose=1, validation_data=(testX,testY),batch_size=batch_size)
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

plt.plot(original, color='red', label='Real Wind Power')
plt.plot(pred, color='blue', label='Predicted Wind Power')
plt.title('Wind Power Prediction')
plt.xlabel('Time')
plt.ylabel('Wind Power')
plt.legend()
plt.show()

model.save('Model_future_value_CNN.h5')
print('Model Saved!')
# ##########evaluate##############
# # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
# mse = mean_squared_error(pred, original)
# # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
# rmse = math.sqrt(mean_squared_error(pred, original))
# # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
# mae = mean_absolute_error(pred, original)
# mape = mean_absolute_percentage_error(pred, original)
# print('均方误差: %.6f' % mse)
# print('均方根误差: %.6f' % rmse)
# print('平均绝对误差: %.6f' % mae)
# print('平均绝对百分误差: %.6f' % mape)
