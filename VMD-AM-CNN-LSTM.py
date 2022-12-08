"""
1.VDM分解
2.分解后分别进行
VMD-LSTM
VMD-AT-LSTM
VMD-CNN-LSTM
VMD-AM-CNN-LSTM预测
3.预测后分别重构，进行MAPE评估
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

timesteps = seq_length = 60

output_dim = 1

data_dim = 8
epochs = 30
batch_size = 32
VMD_num = 5

df = pd.read_csv("./data3/VMDban-5.csv")

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


def bulid_model():
    def attention_3d_block(inputs):
        x = tf.keras.layers.Permute((2, 1))(inputs)
        x = tf.keras.layers.Dense(seq_length, activation="softmax")(x)
        attention_probs = tf.keras.layers.Permute((2, 1), name="attention_vec")(x)
        multipy_layer = tf.keras.layers.Multiply()([input_layer, attention_probs])
        return multipy_layer

    input_layer = tf.keras.Input(shape=(seq_length, data_dim))
    lstm_layer = tf.keras.layers.LSTM(data_dim, return_sequences=True)(input_layer)
    cnn_layer = tf.keras.layers.Conv1D(filters=data_dim, kernel_size=5, padding='same', strides=1, activation='relu',
                                       )(lstm_layer)
    cnn_layer = tf.keras.layers.Conv1D(filters=data_dim, kernel_size=3, padding='same', strides=1, activation='relu',
                                       )(cnn_layer)
    cnn_layer = tf.keras.layers.MaxPool1D(pool_size=2,strides=1, padding='valid'
                                       )(cnn_layer)

    attention_mul = attention_3d_block(cnn_layer)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, return_sequences=True))(attention_mul)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=200))(x)

    attention_mul = tf.keras.layers.Flatten()(x)
    dense = tf.keras.layers.Dense(25)(attention_mul)
    # print("打平后:",attention_mul.shape)
    output = tf.keras.layers.Dense(1)(dense)
    # print('dense',dense.shape)
    model = tf.keras.Model(inputs=[input_layer], outputs=[output])

    model.compile(loss='mse', metrics='mape', optimizer='Adam')
    print(model.summary())
    return model


model = bulid_model()
model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), verbose=1, batch_size=batch_size)

prediction = model.predict(testX)

'''
因为在缩放数据时，我们每行有 14 列，现在我们只有 1 列是目标列。
所以我们必须改变形状来使用 inverse_transform
'''
prediction_copies_array = np.repeat(prediction, data_dim, axis=-1)
'''
14列值是相似的，它只是将单个预测列复制了 4 次。所以现在我们有 5 列相同的值 。
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

plt.plot(original, color='red', label='Real Wind Power(VMD)')
plt.plot(pred, color='blue', label='Predicted Wind Power(VMD)')
plt.title('Wind Power Prediction(VMD)')
plt.xlabel('Time')
plt.ylabel('Wind Power(VMD)')
plt.legend()
plt.show()

model.save('./models/VMD{}_LSTM_CNN_AM_epochs_{}_batch_size_{}.h5'.format(VMD_num, epochs, batch_size))
print('Model Saved!')
