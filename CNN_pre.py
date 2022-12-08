"""
均方误差: 30073.629039
均方根误差: 173.417499
平均绝对误差: 113.845474
平均绝对百分误差: 0.776718
"""
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

pre_model = load_model('Model_future_value_CNN.h5')

df=pd.read_csv("./data2/handle_Turbine_Data .csv")
test_split=round(len(df)*0.20)
df_for_training=df[:-9685]
df_for_testing=df[-9685:]
print(df_for_training.shape)
print(df_for_testing.shape)

'''
可以注意到数据范围非常大，并且它们没有在相同的范围内缩放，
因此为了避免预测错误，让我们先使用MinMaxScaler缩放数据。
(也可以使用StandardScaler)
'''
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
print(df_for_training_scaled)

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)
trainX,trainY=createXY(df_for_training_scaled,60)
testX,testY=createXY(df_for_testing_scaled,60)

# print("trainX Shape-- ",trainX.shape)
# print("trainY Shape-- ",trainY.shape)
# print("testX Shape-- ",testX.shape)
# print("testY Shape-- ",testY.shape)
#
# print("trainX[0]-- \n",trainX[0])
# print("trainY[0]-- ",trainY[0])
prediction=pre_model.predict(testX)
# print("prediction\n", prediction)
# print("\nPrediction Shape-",prediction.shape)
'''
因为在缩放数据时，我们每行有 14 列，现在我们只有 1 列是目标列。
所以我们必须改变形状来使用 inverse_transform
'''
prediction_copies_array = np.repeat(prediction,14, axis=-1)
'''
14列值是相似的，它只是将单个预测列复制了 4 次。所以现在我们有 5 列相同的值 。
'''
# print(prediction_copies_array.shape)
"""
这样就可以使用 inverse_transform 函数。
"""
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),14)))[:,0]

"""
但是逆变换后的第一列是我们需要的，所以我们在最后使用了 → [:,0]。
现在将这个 pred 值与 testY 进行比较，但是 testY 也是按比例缩放的，也需要使用与上述相同的代码进行逆变换。
"""
original_copies_array = np.repeat(testY,14, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),14)))[:,0]

# print("Pred Values-- " ,pred)
# print("\nOriginal Values-- " ,original)

plt.plot(original, color = 'red', label = 'Real Wind Power')
plt.plot(pred, color = 'blue', label = 'Predicted Wind Power')
plt.title('Wind Power Prediction')
plt.xlabel('Time')
plt.ylabel('Wind Power')
plt.legend()
plt.show()
##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(pred, original)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(pred, original))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(pred, original)
mape = mean_absolute_percentage_error(pred, original)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
print('平均绝对百分误差: %.6f' % mape)
