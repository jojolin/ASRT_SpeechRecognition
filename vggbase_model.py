#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""

from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization, Flatten
from tensorflow.keras.layers import Lambda, TimeDistributed, Activation, SeparableConv2D, Conv2D, MaxPooling2D #, Merge
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta, Adam
from readdata24 import DataSpeech

class SpeechModel():

    def __init__(self):
        pass

    def create_model(self, audio_length, audio_feature_length, ms_output_size, label_max_string_length):

        input_data = Input(name='the_input', shape=(audio_length, audio_feature_length, 1))

        depth = 32
        layerx = Conv2D(depth, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
        layerx = Dropout(0.05)(layerx)
        layerx = SeparableConv2D(depth, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(layerx)
        layerx = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layerx) # 池化层

        for depth, pool_size, dropout in zip(
                [64, 128, 128, 128],
                [2,  2,   1,   1],
		[0.1,0.15,0.15,0.2]):
            layerx = Dropout(dropout)(layerx)
            layerx = SeparableConv2D(depth, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(layerx)
            layerx = Dropout(dropout)(layerx)
            layerx = SeparableConv2D(depth, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(layerx)
            layerx = MaxPooling2D(pool_size=pool_size, strides=None, padding="valid")(layerx) # 池化层

        print('layerx', layerx)
        layer_x2 = Reshape((200, 1440))(layerx) #Reshape层
        layer_x2 = Dropout(0.3)(layer_x2) #Reshape层
        layer_x2 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_x2) # 全连接层
        layer_x2 = Dropout(0.3)(layer_x2) #Reshape层
        layer_x2 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_x2) # 全连接层
        layer_x2 = Dense(ms_output_size, use_bias=True, kernel_initializer='he_normal')(layer_x2) # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_x2)
        model_base = Model(inputs = input_data, outputs = y_pred)
        model_base.summary()

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        model.summary()

        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)

        # captures output of softmax so we can decode the output during visualization
        #test_func = K.function([input_data], [y_pred])

        #print('[*提示] 创建模型成功，模型编译成功')
        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        return model_base, model

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

