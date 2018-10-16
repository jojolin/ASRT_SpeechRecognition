#!/usr/bin/env python
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, MaxPooling2D, Reshape, Activation,Flatten, Conv3D, MaxPooling3D, \
    Lambda
from tensorflow.keras.optimizers import SGD, Adadelta, Adam
from tensorflow.keras import backend as K

class SpeechModel(object):

    def __init__(self):

        self.AUDIO_LENGTH = 1200
        self.AUDIO_FEATURE_LENGTH = 100
        self.MS_OUTPUT_SIZE = 100
        self.label_max_string_length = 30

        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data) # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1) # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2) # 池化层
        layer_h3 = Dropout(0.05)(layer_h3)

        layer_h4 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3) # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4) # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层
        layer_h6 = Dropout(0.1)(layer_h6)

        layer_h7 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6) # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7) # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8) # 池化层
        layer_h9 = Dropout(0.15)(layer_h9)

        #layer_h10 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9) # 卷积层
        #layer_h10 = Dropout(0.2)(layer_h10)
        #layer_h11 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10) # 卷积层
        #layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11) # 池化层
        #layer_h12 = Dropout(0.2)(layer_h12)

        #layer_h13 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h12) # 卷积层
        #layer_h13 = Dropout(0.2)(layer_h13)
        #layer_h14 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h13) # 卷积层
        #layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14) # 池化层
        #layer_h15 = Dropout(0.2)(layer_h15)

        layer_h10 = Conv2D(32, (1,1), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9) # 卷积层
        layer_h11 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h10) # 池化层
        print('layer_h11:', layer_h11.shape)
        layer_h12 = Reshape((layer_h11.shape[1], layer_h11.shape[2] * layer_h11.shape[3]))(layer_h11) #Reshape层
        #layer_h12 = Flatten()(layer_h11) #Reshape层

        #layer_h17 = Dense(32, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16) # 全连接层
        #layer_h17 = Dropout(0.3)(layer_h16)

        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h12) # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        # base model
        self.model_base = Model(inputs = input_data, outputs = y_pred)
        self.model_base.summary()

        # ctc model
        labels = Input(name='the_labels', shape=(self.label_max_string_length, ), dtype='float32')
        input_length = Input(name='input_length', shape=(1,), dtype='int64')
        label_length = Input(name='label_length', shape=(1,), dtype='int64')
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        self.model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        self.model.summary()

        #opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        #opt = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        print(y_pred)
        #y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def train(self):
        def data_gen(batch_size=50):
            step = 0
            while step < 20:
                yield [
                        np.random.random((batch_size, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1)),  # X
                        np.random.random((batch_size, self.label_max_string_length)), # data_labels
                        np.array([self.AUDIO_LENGTH]* batch_size).T,  # input_length
                        np.matrix([[self.label_max_string_length]] * batch_size) # label_lenth
                      ], np.array([0.0] * batch_size, dtype=np.float) # labels

        # train
        #data = np.random.random((100, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))
        #labels = np.random.randint(2, size=(100, 128))
        #labels = np.random.randint(2, size=(100, self.label_max_string_length))
        #self.model.train_on_batch(data, labels, self.AUDIO_LENGTH, len(labels))

        self.model.fit_generator(data_gen(), 1000)

        # save
        save_h5 = 'speech.tfkeras.h5'
        self.model.save(save_h5)
        print('model save:', save_h5)

        self.conv_tflite(self.model, 'speech.tfkeras.h5')

    def predict(self, modelfp):
        if modelfp.endswith('.tflite'):
            print('run with tflite')
            run_tflite(modelfp)
        else:
            print('run with tf')
            self.run_tf(modelfp)
        time.sleep(2)

    def run_tf(self,  modelfp):
        self.model.load_weights(modelfp)
        input_shape = (1, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1)
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        output = self.model.predict(input_data)
        print('output:', output)

    def conv_tflite2(self, modelfp):
        self.model.load_weights(modelfp)
        self.conv_tflite(self.model, modelfp)

    def conv_tflite(self, model, save_h5):
        converter = tf.contrib.lite.TocoConverter.from_keras_model_file(save_h5)
        tflite_model = converter.convert()
        outfp = save_h5 + ".tflite"
        print(outfp)
        open(outfp, 'wb').write(tflite_model)

def run_tflite(tflite_model):
    interpreter = tf.contrib.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('input_detail', input_details)
    print('output_detail', output_details)

    input_shape = input_details[0]['shape']
    print('input_shape:', input_shape)
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

def main():
    tp = sys.argv[1]
    if tp == 't':
        model = SpeechModel()
        print('train model')
        model.train()
    elif tp == 'r':
        model = SpeechModel()
        print('run model', sys.argv[2])
        model.predict(sys.argv[2])
    elif tp == 'c':
        model = SpeechModel()
        model.conv_tflite2(sys.argv[2])

if __name__ == '__main__':
    main()
