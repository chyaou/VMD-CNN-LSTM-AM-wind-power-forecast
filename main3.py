"""
2022.7.2 15:50
author:chy
在main2的基础上使用hyperopt进行参数优化
"""
import numpy as np
import tensorflow
import tensorflow as tf
from hyperopt import hp, Trials, fmin, tpe
from hyperopt.early_stop import no_progress_loss
from tensorflow.python.keras.layers import Dropout, Dense, LSTM
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 定义参数模型空间
params_space = {
    "units": hp.quniform('units', 40, 45, 4),
    "epochs": hp.quniform('epochs', 1, 2, 1),
    "loss": hp.choice("loss", ["mae", "mse"])
}


def model(params):
    model = tf.keras.Sequential()
    model.add(LSTM(units=int(params['units'])))
    model.add(Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=str(params['loss']),
                  )  # 损失函数用均方误差
    # # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
    # checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt"
    #
    # if os.path.exists(checkpoint_save_path + '.index'):
    #     print('-------------load the model-----------------')
    #     model.load_weights(checkpoint_save_path)

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
    #                                                  save_weights_only=False,
    #                                                  save_best_only=True,
    #                                                  monitor='val_loss')
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x_train, y_train, batch_size=64, epochs=int(params['epochs']), validation_data=(x_test, y_test))
    model.summary()
    # file = open('./weights.txt', 'w')  # 参数提取
    # for v in model.trainable_variables:
    #     file.write(str(v.name) + '\n')
    #     file.write(str(v.shape) + '\n')
    #     file.write(str(v.numpy()) + '\n')
    # file.close()
    # 用测试集上的损失进行寻优
    score_loss = model.evaluate(x_test, y_test, verbose=2)
    print("Text loss:", score_loss)
    return score_loss


# 定义参数优化函数
def param_hyperopt(max_evals=100):
    trials = Trials()
    # 提前停止条件
    early_stop_fn = no_progress_loss(20)
    # 优化模型
    params_best = fmin(fn=model, space=params_space, algo=tpe.suggest, max_evals=max_evals,
                       trials=trials, early_stop_fn=early_stop_fn)
    print('best params:', params_best)
    return params_best, trials


wind_power = pd.read_csv('data2/handle_Turbine_Data.csv')  # 读取风力功率文件

training_set = wind_power.iloc[0:32284 - 6000,
               0:1].values  # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
test_set = wind_power.iloc[32284 - 6000:, 0:1].values  # 后300天的开盘价作为测试集

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化

x_train = []
y_train = []

x_test = []
y_test = []

# 测试集：csv表格中前2426-300=2126天数据
# 利用for循环，遍历整个训练集，提取训练集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建2426-300-60=2066组数据。
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)

# 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]即2066组数据；输入60个开盘价，预测出第61天的开盘价，循环核时间展开步数为60; 每个时间步送入的特征是某一天的开盘价，只有1个数据，故每个时间步输入特征个数为1
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
# 测试集：csv表格中后300天数据
# 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建300-60=240组数据。
for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])
    y_test.append(test_set[i, 0])
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

params_best, trials = param_hyperopt(3)
print(trials.results)
print(trials.losses())
print(trials.statuses)
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# mape_plt = history.history['mape']
# val_mape_plt = history.history['val_mape']
#
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
# plt.plot(mape_plt, label='Training mape')
# plt.plot(val_mape_plt, label='Validation mape')
# plt.title('Training and Validation mape')
# plt.legend()
# plt.show()
# ################## predict ######################
# # 测试集输入模型进行预测
# predicted_stock_price = model.predict(x_test)
# # 对预测数据还原---从（0，1）反归一化到原始范围
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# # 对真实数据还原---从（0，1）反归一化到原始范围
# real_stock_price = sc.inverse_transform(test_set[60:])
# # 画出真实数据和预测数据的对比曲线
# plt.plot(real_stock_price, color='red', label='Wind Power')
# plt.plot(predicted_stock_price, color='blue', label='Predicted Wind Power')
# plt.title('Wind Power Prediction')
# plt.xlabel('Time')
# plt.ylabel('Wind Power')
# plt.legend()
# plt.show()
#
# ##########evaluate##############
# # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
# mse = mean_squared_error(predicted_stock_price, real_stock_price)
# # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
# rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
# mae = mean_absolute_error(predicted_stock_price, real_stock_price)
# mape = mean_absolute_percentage_error(predicted_stock_price, real_stock_price)
# print('均方误差: %.6f' % mse)
# print('均方根误差: %.6f' % rmse)
# print('平均绝对误差: %.6f' % mae)
# print('平均绝对百分误差: %.6f' % mape)
