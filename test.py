import tensorflow as tf

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Permute
from tensorflow.python.keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

start = time.time()
timesteps = seq_length = 60
data_dim = 14
output_dim = 1

df = pd.read_csv("./data2/handle_Turbine_Data .csv")
test_split = round(len(df) * 0.20)
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

def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)
trainX, trainY = createXY(df_for_training_scaled, 60)
testX, testY = createXY(df_for_testing_scaled, 60)
def bulid_model(optimizer):
    def attention_3d_block(inputs):
        x = tf.keras.layers.Permute((2, 1))(inputs)
        x = tf.keras.layers.Dense(seq_length, activation="softmax")(x)
        attention_probs = tf.keras.layers.Permute((2, 1), name="attention_vec")(x)
        multipy_layer = tf.keras.layers.Multiply()([input_layer, attention_probs])
        return multipy_layer

    input_layer = tf.keras.Input(shape=(seq_length, data_dim))
    lstm_layer = tf.keras.layers.LSTM(data_dim, return_sequences=True)(input_layer)
    cnn_layer = tf.keras.layers.Conv1D(filters=14, kernel_size=3, padding='same', strides=1, activation='relu',
                                       )(lstm_layer)

    attention_mul = attention_3d_block(cnn_layer)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, return_sequences=True))(attention_mul)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=200))(x)

    attention_mul = tf.keras.layers.Flatten()(x)
    dense = tf.keras.layers.Dense(100)(attention_mul)
    # print("打平后:",attention_mul.shape)
    output = tf.keras.layers.Dense(1)(dense)
    # print('dense',dense.shape)
    model = tf.keras.Model(inputs=[input_layer], outputs=[output])

    model.compile(loss='mse', metrics='mape', optimizer=optimizer)
    print(model.summary())
    return model





strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量



with strategy.scope():
    grid_model = KerasRegressor(build_fn=bulid_model, verbose=1, validation_data=(testX, testY))
    parameters = {'batch_size': [32, 64, 128],
                  'epochs': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  'optimizer': ['adam', 'RMSProp']}
    grid_search = GridSearchCV(estimator=grid_model,
                               param_grid=parameters,
                               cv=2)

    grid_search = grid_search.fit(trainX, trainY)
    print(grid_search.best_params_)
    my_model = grid_search.best_estimator_.model

prediction = my_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-", prediction.shape)
'''
因为在缩放数据时，我们每行有 14 列，现在我们只有 1 列是目标列。
所以我们必须改变形状来使用 inverse_transform
'''
prediction_copies_array = np.repeat(prediction, 14, axis=-1)
'''
14列值是相似的，它只是将单个预测列复制了 4 次。所以现在我们有 5 列相同的值 。
'''
print(prediction_copies_array.shape)
"""
这样就可以使用 inverse_transform 函数。
"""
pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), 14)))[:, 0]

"""
但是逆变换后的第一列是我们需要的，所以我们在最后使用了 → [:,0]。
现在将这个 pred 值与 testY 进行比较，但是 testY 也是按比例缩放的，也需要使用与上述相同的代码进行逆变换。
"""
original_copies_array = np.repeat(testY, 14, axis=-1)
original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), 14)))[:, 0]

print("Pred Values-- ", pred)
print("\nOriginal Values-- ", original)

print('Model Saved!')
from tip import send_message

t = time.time() - start
send_message('代码处理好了，用时{}min'.format(str(t / 60)))
send_message('参数：{}'.format(str(grid_search.best_params_)))