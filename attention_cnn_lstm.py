import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.layers import Layer

file_path = "./Volumes/Elements/kaggle/AT-LSTM/metadata_test.csv"

df = pd.read_csv(file_path)

plt.figure()
plt.plot(range(df.shape[0]), (df['NDX']))
plt.xlabel('time', fontsize=18)
plt.ylabel('NDX', fontsize=18)
plt.show()

BATCH_SIZE = 128
EPOCHS = 3
SEQ_LEN = 10
FUTURE_PERIOD_PREDICT = 1
RATIO_TO_PREDICT = "NDX"
targ_cols = ("NDX",)


# 归一化，划分feature target
def preprocess_data(dat, col_names):
    scale = MinMaxScaler().fit(dat)
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]

    return feats, targs


# 把数据切分成80%训练集、20%测试集
def split_data(data, percent_train=0.70):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        if idx < num_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)
    return np.array(train_data), np.array(test_data)


data_X, data_y = preprocess_data(df, targ_cols)
print(data_X.shape, data_y.shape)
all_data = np.concatenate([data_X, data_y], axis=1)
print(all_data.shape)

train_, test_ = split_data(all_data)
print("train shape {0}".format(train_.shape))
print("test shape {0}".format(test_.shape))


def timestamp_data(data):
    X = []
    Y = []
    for i in range(SEQ_LEN, len(data) - FUTURE_PERIOD_PREDICT + 1):
        X.append(data[i - SEQ_LEN:i, :data_X.shape[1]])  # 含左不含右
        Y.append(data[i + (FUTURE_PERIOD_PREDICT - 1), data_X.shape[1]])
    return np.array(X), np.array(Y)


X_train, y_train = timestamp_data(train_)
print('X_train', X_train.shape, 'y_train', y_train.shape)
X_test, y_test = timestamp_data(test_)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], data_X.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], data_X.shape[1]))

print("train shape {0}".format(X_train.shape))
# print("valid shape {0}".format(X_valid.shape))
print("test shape {0}".format(X_test.shape))


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')
        # W_regularizer: 权重上的正则化
        # b_regularizer: 偏置项的正则化
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
        # W_constraint: 权重上的约束项
        # b_constraint: 偏置上的约束项
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        '''
        keras.backend.cast(x, dtype): 将张量转换到不同的 dtype 并返回
        '''
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        '''
        keras.backend.epsilon(): 返回浮点数
        '''
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


inp = Input(shape=(SEQ_LEN, data_X.shape[1]))
x = GRU(256, return_sequences=True)(inp)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Attention(SEQ_LEN)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1, activation="relu")(x)
model_lstm_attention = tf.keras.Model(inputs=inp, outputs=x)
model_lstm_attention.compile(loss='mean_squared_error', optimizer='adam')
model_lstm_attention.summary()

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

tf.keras.utils.plot_model(model_lstm_attention,
                          to_file="model_lstm_attention.png",
                          show_shapes=True)

model_lstm_attention.fit(X_train, y_train,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS,
                         validation_data=(X_test, y_test))

# model_lstm_attention.save('lstm+gru.h5')

predicted_LSTM_Att = model_lstm_attention.predict(X_test)

# 绘图
plt.plot(list(range(X_test.shape[0])), y_test, label='True')
plt.plot(list(predicted_LSTM_Att), label='Predicted Test')
plt.title('ffmpeg-power')
plt.xlabel('DateTime')
plt.ylabel('power')
plt.legend()
plt.savefig("/Volumes/Elements/kaggle/AT-LSTM/results")
plt.show()
