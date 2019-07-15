# -*- coding:utf-8 -*-
# Author:Chen ke yu
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten


class RegModel(object):
    def __init__(self):
        return

    @staticmethod
    def Model_Easy_LSTM(look_back=20):
        model = Sequential()
        model.add(LSTM(units=40, input_shape=(1, look_back)))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @staticmethod
    def Model_DNN_LSTM(look_back=20):
        model = Sequential()
        model.add(LSTM(units=80, input_shape=(1, look_back)))
        model.add(Dense(units=20))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @staticmethod
    def Model_LR(input_size=20):
        model = Sequential()
        model.add(Flatten(input_shape=(1, 20)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @staticmethod
    def Model_DNN():
        model = Sequential()
        model.add(Flatten(batch_input_shape=(None, 1, 20)))
        model.add(Dense(units=40))
        model.add(Dense(units=20))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

