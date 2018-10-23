#!/usr/bin/env python
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, SeparableConv2D, Lambda, Reshape, Activation, AvgPool2D, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class SpeechModel(object):
    '''
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
    '''
    def init_model(self):
        self.model_base, self.model = self.create_model(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, self.MS_OUTPUT_SIZE)

    def create_model(self, audio_length, audio_feature_length, ms_output_size, label_max_string_length):
        input_data = Input(name='the_input', shape=(audio_length, audio_feature_length, 1))
        layer_x = Conv2D(32, (3,3), 1, use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data)

        for depth, stride in zip(
                [64, 64, 128, 128, 256, 256],
                [2,  1,   2,   1,   2,   1]):
            layer_x = SeparableConv2D(depth, (3,3), stride, use_bias=True, activation='relu', padding='same')(layer_x)
            #layer_x = Conv2D(depth, (1,1), 1, use_bias=True, activation='relu', padding='same')(layer_x) # 卷积层

        #layer_x2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_x)
        layer_x2 = layer_x
        print('layer_x:', layer_x.shape)
        layer_x2 = Reshape((layer_x2.shape[1], layer_x2.shape[2]* layer_x2.shape[3]))(layer_x2) #Reshape层

        layer_x2 = Dense(ms_output_size, use_bias=True, kernel_initializer='he_normal')(layer_x2) # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_x2)

        # base model
        model_base = Model(inputs = input_data, outputs = y_pred)
        model_base.summary()

        # ctc model
        labels = Input(name='the_labels', shape=(label_max_string_length, ), dtype='float32')
        input_length = Input(name='input_length', shape=(1,), dtype='int64')
        label_length = Input(name='label_length', shape=(1,), dtype='int64')
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        model.summary()

        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)

        return model_base, model

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        #y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

