# 0
"""
ARIMA
均方误差: 295645.363571
均方根误差: 543.732805
平均绝对误差: 480.871956
平均绝对百分误差: 0.692612
"""
# 1
"""
VMD-LSTM重构
均方误差: 19913.511135
均方根误差: 141.115241
平均绝对误差: 119.465079
平均绝对百分误差: 0.393829
"""
# 2
"""
VMD-CNN-LSTM重构
均方误差: 30366.376248
均方根误差: 174.259508
平均绝对误差: 144.900007
平均绝对百分误差: 0.408477
"""
# 3
"""
VMD-AM-CNN-LSTM重构 dense = 50
均方误差: 10563.884262
均方根误差: 102.780758
平均绝对误差: 75.007315
平均绝对百分误差: 0.602590
"""
"""
VMD-AM-CNN-LSTM重构 dense = 25 2022.9.26
发现问题：经过多次重构，发现第四次重构比五次全部重构要好一些（2022.9.26）
均方误差: 13107.831739 
均方根误差: 114.489439
平均绝对误差: 89.132676
平均绝对百分误差: 0.382869
"""
# 4
"""
VMD-CNN重构 2022.9.26
均方误差: 25241.065831
均方根误差: 158.874371
平均绝对误差: 132.292444
平均绝对百分误差: 0.394348
"""
# 5
"""
VMD-AM-LSTM重构 dense = 25 2022.9.27
均方误差: 10726.633921
均方根误差: 103.569464
平均绝对误差: 74.672946
平均绝对百分误差: 0.686071

"""
# 6 svm
"""
均方误差: 0.135460
均方根误差: 0.368049
平均绝对误差: 0.119101
平均绝对百分误差: 0.021031
"""
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
predict_num = 1000
data_dim = 8
dic1 = {'./models/VMD1_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-1.csv',
        './models/VMD2_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-2.csv',
        './models/VMD3_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-3.csv',
        './models/VMD4_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-4.csv',
        './models/VMD5_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-5.csv',
        }
dic2 = {'./models/VMD1_CNN_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-1.csv',
        './models/VMD2_CNN_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-2.csv',
        './models/VMD3_CNN_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-3.csv',
        './models/VMD4_CNN_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-4.csv',
        './models/VMD5_CNN_LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-5.csv',
        }
dic3 = {'./models/VMD1_LSTM_CNN_AM_epochs_30_batch_size_32.h5': 'data3/VMDban-1.csv',
        './models/VMD2_LSTM_CNN_AM_epochs_30_batch_size_32.h5': 'data3/VMDban-2.csv',
        './models/VMD3_LSTM_CNN_AM_epochs_30_batch_size_32.h5': 'data3/VMDban-3.csv',
        './models/VMD4_LSTM_CNN_AM_epochs_30_batch_size_32.h5': 'data3/VMDban-4.csv',
        './models/VMD5_LSTM_CNN_AM_epochs_30_batch_size_32.h5': 'data3/VMDban-5.csv',
        }
dic4 = {'./models/VMD1_CNN_epochs_30_batch_size_32.h5': 'data3/VMDban-1.csv',
        './models/VMD2_CNN_epochs_30_batch_size_32.h5': 'data3/VMDban-2.csv',
        './models/VMD3_CNN_epochs_30_batch_size_32.h5': 'data3/VMDban-3.csv',
        './models/VMD4_CNN_epochs_30_batch_size_32.h5': 'data3/VMDban-4.csv',
        './models/VMD5_CNN_epochs_30_batch_size_32.h5': 'data3/VMDban-5.csv',
        }
dic5 = {'./models/VMD1_AM-LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-1.csv',
        './models/VMD2_AM-LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-2.csv',
        './models/VMD3_AM-LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-3.csv',
        './models/VMD4_AM-LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-4.csv',
        './models/VMD5_AM-LSTM_epochs_30_batch_size_32.h5': 'data3/VMDban-5.csv',
        }


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


def read_data_(model_name, file_name):
    pre_model = load_model(model_name)
    df = pd.read_csv(file_name)
    return pre_model, df


def deal_data_(model_name, file_name):
    pre_model, df = read_data_(model_name, file_name)

    df_for_testing = df[-13000:-12000]
    scaler = MinMaxScaler(feature_range=(0, 1))

    df_for_testing_scaled = scaler.fit_transform(df_for_testing)

    testX, testY = createXY(df_for_testing_scaled, 60)

    prediction = pre_model.predict(testX)
    prediction_copies_array = np.repeat(prediction, data_dim, axis=-1)
    pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), data_dim)))[:, 0]

    return pred


def plot_(model_name, file_name):
    pred, original = deal_data_(model_name, file_name)

    plt.plot(original, color='red', label='Real Wind Power')
    plt.plot(pred, color='blue', label='Predicted Wind Power')
    plt.title('Wind Power Prediction')
    plt.xlabel('Time')
    plt.ylabel('Wind Power')
    plt.legend()
    plt.show()


def evaluation_(pred, original):
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


if __name__ == '__main__':
    file_original = 'data2/handle_Turbine_Data .csv'
    df = pd.read_csv(file_original)
    df_for_testing = df[-13000:-12000]
    # df_for_testing = df[-8000:-7000]  # 夏季
    # df_for_testing = df[-18000:-17000] # 冬季
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_for_testing_scaled = scaler.fit_transform(df_for_testing)
    testX, testY = createXY(df_for_testing_scaled, 60)
    original_copies_array = np.repeat(testY, data_dim, axis=-1)
    original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), data_dim)))[:, 0]

    pred1 = [0] * predict_num
    for model_name, file_name in dic1.items():
        pred1 = list(map(lambda m, n: m + n, pred1, deal_data_(model_name, file_name)))
    evaluation_(pred1,original)

    pred2 = [0] * predict_num
    for model_name, file_name in dic2.items():
        pred2 = list(map(lambda m, n: m + n, pred2, deal_data_(model_name, file_name)))
    evaluation_(pred2, original)

    pred3 = [0] * predict_num
    for model_name, file_name in dic3.items():
        pred3 = list(map(lambda m, n: m + n, pred3, deal_data_(model_name, file_name)))
    evaluation_(pred3, original)

    pred4 = [0] * predict_num
    for model_name, file_name in dic4.items():
        pred4 = list(map(lambda m, n: m + n, pred4, deal_data_(model_name, file_name)))
    evaluation_(pred4, original)

    pred5 = [0] * predict_num
    for model_name, file_name in dic5.items():
        pred5 = list(map(lambda m, n: m + n, pred5, deal_data_(model_name, file_name)))
    evaluation_(pred5, original)
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    plt.plot(original, color='red', label='实际值',marker='.',markersize=2)
    plt.plot(pred1, color='blue', label='VMD-LSTM',marker='o',markersize=2)
    plt.plot(pred2, color='yellow', label='VMD-CNN-LSTM',marker='v',markersize=2)

    plt.plot(pred4, color='darkblue', label='VMD-CNN',marker='*',markersize=2)
    plt.plot(pred5, color='green', label='VMD-AM-LSTM',marker='+',markersize=2)
    plt.plot(pred5, color='purple', label='VMD-LSTM-CNN-AM', marker='s', markersize=2)
    plt.tick_params(labelsize=20)
    plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.xlabel('采样点', fontproperties='Simsun', fontsize=20)
    plt.ylabel('风电功率/kW', fontproperties='Simsun', fontsize=20)
    # plt.tight_layout()
    plt.legend(loc=2, prop={'size': 20})
    plt.show()
