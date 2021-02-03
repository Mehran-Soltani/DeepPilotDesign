from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datetime import datetime
from os.path import join, exists
from os import makedirs
import math
from keras import backend as K
from keras import Model
from keras.layers import Layer, Softmax, Input
from keras.callbacks import EarlyStopping
from keras.initializers import Constant, glorot_normal
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from keras.models import Sequential,  Model
from keras.layers import Convolution2D,Input,BatchNormalization,Conv2D,Activation,Lambda,Subtract,Conv2DTranspose, PReLU
from keras.regularizers import l2
from keras.layers import  Reshape,Dense,Flatten
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy
import math
import scipy.io
import numpy as np


class ConcreteSelect(Layer):

    def __init__(self, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)

    def build(self, input_shape):
        self.temp = self.add_weight(name='temp', shape=[], initializer=Constant(self.start_temp), trainable=False)
        self.logits = self.add_weight(name='logits', shape=[self.output_dim, input_shape[1]],
                                      initializer=glorot_normal(), trainable=True)
        super(ConcreteSelect, self).build(input_shape)

    def call(self, X, training=None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (self.logits + gumbel) / temp
        samples = K.softmax(noisy_logits)

        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])

        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))

        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class StopperCallback(EarlyStopping):

    def __init__(self, mean_max_target=0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor='', patience=float('inf'), verbose=1, mode='max',
                                              baseline=self.mean_max_target)

    def on_epoch_begin(self, epoch, logs=None):
        print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature',
              K.get_value(self.model.get_layer('concrete_select').temp))
        # print( K.get_value(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        # print(K.get_value(K.max(self.model.get_layer('concrete_select').selections, axis = -1)))

    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis=-1)))
        return monitor_value


class ConcreteAutoencoderFeatureSelector():

    def __init__(self, K, output_function, num_epochs=300, batch_size=None, learning_rate=0.001, start_temp=10.0,
                 min_temp=0.1, tryout_limit=1):
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit

    def fit(self, X, Y=None, val_X=None, val_Y=None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)

        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)

        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size

        for i in range(self.tryout_limit):

            K.set_learning_phase(1)

            inputs = Input(shape=X.shape[1:])

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))

            self.concrete_select = ConcreteSelect(self.K, self.start_temp, self.min_temp, alpha, name='concrete_select')

            selected_features = self.concrete_select(inputs)

            outputs = self.output_function(selected_features)

            self.model = Model(inputs, outputs)

            self.model.compile(Adam(self.learning_rate), loss='mean_squared_error')

            print(self.model.summary())

            stopper_callback = StopperCallback()

            hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose=1, callbacks=[stopper_callback],
                                  validation_data=validation_data)  # , validation_freq = 10)

            if K.get_value(K.mean(
                    K.max(K.softmax(self.concrete_select.logits, axis=-1)))) >= stopper_callback.mean_max_target:
                break

            num_epochs *= 2

        self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

        return self

    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits),
                                           self.model.get_layer('concrete_select').logits.shape[1]), axis=0))

    def transform(self, X):
        return X[self.get_indices()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model


def load_channel(num_pilots, SNR):
    # perfect = loadmat("./data/Perfect_H_40000.mat")["My_perfect_H"]
    perfect = loadmat("./VehA_perfect_all.mat")["H_p_rearranged"]
    perfect = np.transpose(perfect, [2, 0, 1])
    print(perfect.shape)
    perfect_image = np.zeros((len(perfect), 72, 14, 2))

    perfect_image[:, :, :, 0] = np.real(perfect)
    perfect_image[:, :, :, 1] = np.imag(perfect)
    perfect_image = np.concatenate((perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]), axis=0).reshape(
        2 * len(perfect), 72, 14, 1)

    perfect_image = perfect_image.squeeze()
    perfect_image = perfect_image.reshape(
        (perfect_image.shape[0], np.dot(perfect_image.shape[1], perfect_image.shape[2])))

    # noisy = loadmat("./data/My_noisy_H_" + str(SNR) + ".mat")["My_noisy_H"]
    noisy = loadmat("./VehA_noisy_all.mat")["H_p_noisy"]
    noisy = np.transpose(noisy, [2, 0, 1])
    print(noisy.shape)
    noisy_image = np.zeros((len(noisy), 72, 14, 2))

    noisy_image[:, :, :, 0] = np.real(noisy)
    noisy_image[:, :, :, 1] = np.imag(noisy)
    noisy_image = np.concatenate((noisy_image[:, :, :, 0], noisy_image[:, :, :, 1]), axis=0).reshape(
        2 * len(noisy), 72, 14, 1)

    noisy_image = noisy_image.squeeze()
    noisy_image = noisy_image.reshape(
        (noisy_image.shape[0], np.dot(noisy_image.shape[1], noisy_image.shape[2])))
    # perfect_image = np.random.uniform(low = 0.01 , high= 0.99 , size = (perfect_image.shape[0], 72*14))
    print(perfect_image.shape)
    print(noisy_image.shape)

    train_data, test_data, train_label, test_label = train_test_split(noisy_image, perfect_image, test_size=1 / 9,
                                                                      random_state=1)
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=1 / 8,
                                                                    random_state=1)
    return (train_data, train_label), (val_data, val_label), (test_data, test_label)


def unif_ind(num_pilots):
    """
    Uniform pilot indices are specified here based on IEEE standard


    :param num_pilots:
    :return:
    """
    if (num_pilots == 48):
        idx_unif = [14 * i for i in range(1, 72, 6)] + [4 + 14 * (i) for i in range(4, 72, 6)] + [7 + 14 * (i) for i in
                                                                                                  range(1, 72, 6)] + [
                       11 + 14 * (i) for i in range(4, 72, 6)]
    elif (num_pilots == 16):
        idx_unif = [4 + 14 * (i) for i in range(1, 72, 9)] + [9 + 14 * (i) for i in range(4, 72, 9)]
    elif (num_pilots == 24):
        idx_unif = [14 * i for i in range(1, 72, 9)] + [6 + 14 * i for i in range(4, 72, 9)] + [11 + 14 * i for i in
                                                                                                range(1, 72, 9)]
    elif (num_pilots == 8):
        idx_unif = [4 + 14 * (i) for i in range(5, 72, 18)] + [9 + 14 * (i) for i in range(8, 72, 18)]
    elif (num_pilots == 36):
        idx_unif = [14 * (i) for i in range(1, 72, 6)] + [6 + 14 * (i) for i in range(4, 72, 6)] + [11 + 14 * i for i in
                                                                                                    range(1, 72, 6)]

    return idx_unif


def interpolate_model(x):
    x = Dense(150)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(780)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(1008)(x)
    return x


def interpolate_train(train_data, train_label, val_data, val_label, num_epochs, batch_size, learning_rate, num_pilots,
                      SNR,
                      type_ind):
    inputs = Input(shape=train_data.shape[1:])
    outputs = interpolate_model(inputs)
    model = Model(inputs, outputs)
    model.compile(Adam(learning_rate), loss='mean_squared_error')
    print(model.summary())
    # stopper_callback = StopperCallback()

    hist = model.fit(train_data, train_label, batch_size, num_epochs, verbose=1,
                     validation_data=(val_data, val_label))  # , validation_freq = 10)

    # model.save_weights(
    # "./interp_weights/interp_" + str(num_pilots) + "_" + str(SNR) + type_ind + ".h5")

    model.save_weights(
        "./interp_weights/interp_" + str(num_pilots) + "_" + "all" + type_ind + ".h5")


def interpolate_predict(test_data, test_label, num_pilots, SNR, type_ind):
    inputs = Input(shape=test_data.shape[1:])
    outputs = interpolate_model(inputs)

    sr_model = Model(inputs, outputs)

    sr_model.load_weights(
        "./interp_weights/interp_" + str(num_pilots) + "_" + "all" + type_ind + ".h5")

    predicted = sr_model.predict(test_data)
    mse = mean_squared_error(predicted, test_label)
    return predicted, mse


def SRCNN_model():
    input_shape = (72, 14, 1)
    x = Input(shape=input_shape)
    c1 = Convolution2D(64, 9, 9, activation='relu', init='he_normal', border_mode='same')(x)
    c2 = Convolution2D(32, 1, 1, activation='relu', init='he_normal', border_mode='same')(c1)
    c3 = Convolution2D(1, 5, 5, init='he_normal', border_mode='same')(c2)

    x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(c3)
    x1 = Activation('relu')(x1)
    # 15 layers, Conv+BN+relu
    for i in range(5):
        x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
        x1 = BatchNormalization(axis=-1, epsilon=1e-3)(x1)
        x1 = Activation('relu')(x1)
        # last layer, Conv
    x1 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
    c4 = Subtract()([c3, x1])  # input - noise

    model = Model(input=x, output=c4)
    ##compile
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def SRCNN_predict_model():
    input_shape = (72, 14, 1)
    x = Input(shape=input_shape)
    c1 = Convolution2D(64, 9, 9, activation='relu', init='he_normal', border_mode='same')(x)
    c2 = Convolution2D(32, 1, 1, activation='relu', init='he_normal', border_mode='same')(c1)
    c3 = Convolution2D(1, 5, 5, init='he_normal', border_mode='same')(c2)

    x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(c3)
    x1 = Activation('relu')(x1)
    # 15 layers, Conv+BN+relu
    for i in range(5):
        x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
        x1 = BatchNormalization(axis=-1, epsilon=1e-3)(x1)
        x1 = Activation('relu')(x1)
        # last layer, Conv
    x1 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
    c4 = Subtract()([c3, x1])  # input - noise

    model = Model(input=x, output=c4)
    ##compile
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def SRCNN_train(train_data, train_label, val_data, val_label, num_epochs, num_pilots, SNR, type):
    srcnn_model = SRCNN_model()
    print(srcnn_model.summary())

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(train_data, train_label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs=num_epochs, verbose=0)

    srcnn_model.save_weights("./SRCNN_weights/SR_Veh_" + str(num_pilots) + "all" + type + ".h5")


def SRCNN_predict(test_data, test_label, num_pilots, SNR, type):
    srcnn_model = SRCNN_predict_model()
    srcnn_model.load_weights("./SRCNN_weights/SR_Veh_" + str(num_pilots) + "all" + type + ".h5")
    predicted = srcnn_model.predict(test_data)
    mse = mean_squared_error(predicted.reshape(predicted.shape[0], 1008), test_label.reshape(test_label.shape[0], 1008))

    return predicted, mse
