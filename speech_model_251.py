#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

from general_function.file_wav import read_wav_data, GetMfccFeature
from general_function.gen_func import GetEditDistance

# LSTM_CNN
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization, Flatten
from tensorflow.keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta, Adam

from dataset_manager import DataSetManager
import config

abspath = ''
ModelName='251'

import mobilebase_model
#import vggbase_model

class ModelSpeech(): # 语音模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出的拼音的表示大小是1422，即1421个拼音+1个空白块
        '''
        self.MS_OUTPUT_SIZE = 1423 #拼音1421 + 1个特殊字符(为了处理音频数据需要) + 1个空白块
        self.label_max_string_length = config.LABEL_LENGTH
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = config.AUDIO_FEATURE_LENGTH
        self.datasetmanager = DataSetManager()
        speechmodel = mobilebase_model.SpeechModel()
        #speechmodel = vggbase_model.SpeechModel()
        self.base_model, self._model = speechmodel.create_model(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, self.MS_OUTPUT_SIZE, self.label_max_string_length)

        self.datapath = datapath

    def TrainModel(self, datapath, epoch = 2, save_step = 1000, batch_size = 32, filename = abspath + 'model_speech/m' + ModelName + '/speech_model'+ModelName):
        '''
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        '''
        yielddatas = self.datasetmanager.data_generator(batch_size, self.AUDIO_LENGTH)

        for epoch in range(epoch): # 迭代轮数
            print('[running] train epoch %d .' % epoch)
            n_step = 0 # 迭代数据数
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+'%(epoch, n_step*save_step))
                    self._model.fit_generator(yielddatas, save_step)
                    n_step += 1
                except StopIteration:
                    print('[error] generator error. please check data format.')
                    break

                self.SaveModel(comment='_e_'+str(epoch)+'_step_'+str(n_step * save_step))
                self.TestModel(data_count=8)
                self.TestModel(data_count=8)

    def LoadModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName+'.model'):
        '''
        加载模型参数
        '''
        self._model.load_weights(filename)
        self.base_model.load_weights(filename + '.base')

    def SaveModel(self, filename=abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName, comment=''):
        '''
        保存模型参数
        '''
        self._model.save_weights(filename+comment+'.model')
        self._model.save(filename+comment+'.h5')
        self.base_model.save_weights(filename + comment + '.model.base')
        self.base_model.save(filename+comment+'.base.h5')
        f = open('step'+ModelName+'.txt','w')
        f.write(filename+comment)
        f.close()

    def TestModel(self, data_count = 32, io_step_print = 10, io_step_file = 10):
        '''
        测试检验模型效果
        io_step_print
            为了减少测试时标准输出的io开销，可以通过调整这个参数来实现
        io_step_file
            为了减少测试时文件读写的io开销，可以通过调整这个参数来实现
        '''
        #num_data = data.GetDataNum() # 获取数据的数量
        words_num = 0
        word_error_num = 0

        for x in range(0, data_count):
            test_data = self.datasetmanager.next_data()
            data_input, data_labels = test_data
            #data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据

            # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
            if data_input.shape[0] > self.AUDIO_LENGTH:
                continue

            pre = self.Predict(data_input, data_input.shape[0] // 8)
            words_n = data_labels.shape[0] # 获取每个句子的字数
            words_num += words_n # 把句子的总字数加上
            edit_distance = GetEditDistance(data_labels, pre) # 获取编辑距离
            if(edit_distance <= words_n): # 当编辑距离小于等于句子字数时
                word_error_num += edit_distance # 使用编辑距离作为错误字数
            else: # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                word_error_num += words_n # 就直接加句子本来的总字数就好了

        print('*[Test Result] Speech Recognition set word error ratio: ', word_error_num / words_num * 100, '%')

    def Predict(self, data_input, input_len):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''
        batch_size = 1
        in_len = np.zeros((batch_size),dtype = np.int32)
        in_len[0] = input_len
        x_in = np.zeros((batch_size, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
        for i in range(batch_size):
            x_in[i,0:len(data_input)] = data_input
        base_pred = self.base_model.predict(x = x_in)
        base_pred =base_pred[:, :, :]
        r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1)
        r1 = K.get_value(r[0][0])
        r1=r1[0]
        return r1

    def RecognizeSpeech(self, wavsignal, fs):
        '''
        最终做语音识别用的函数，识别一个wav序列的语音
        不过这里现在还有bug
        '''
        # 获取输入特征
        data_input = GetMfccFeature(wavsignal, fs, config.AUDIO_MFCC_FEATURE_LENGTH)
        #data_input = GetFrequencyFeature3(wavsignal, fs)
        input_length = len(data_input)
        input_length = input_length // 8

        data_input = np.array(data_input, dtype = np.float)
        #print(data_input,data_input.shape)
        data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
        #t2=time.time()
        r1 = self.Predict(data_input, input_length)
        #t3=time.time()
        #print('time cost:',t3-t2)
        list_symbol_dic = self.datasetmanager.list_symbol # 获取拼音列表
        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])

        return r_str

    def RecognizeSpeech_FromFile(self, *fps):
        '''
        最终做语音识别用的函数，识别指定文件名的语音
        '''
        res = []
        for filename in fps:
            wavsignal,fs = read_wav_data(filename)
            print('read time: ', time.time())
            r = self.RecognizeSpeech(wavsignal, fs)
            print('reco time: ', time.time())
            res.append(r)
        return res

    @property
    def model(self):
        '''
        返回keras model
        '''
        return self._model
