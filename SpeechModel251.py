#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
import platform as plat
import os
import time

from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *

# LSTM_CNN
import numpy as np
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization, Flatten
from tensorflow.keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta, Adam

from readdata24 import DataSpeech
import config

abspath = ''
ModelName='251'

import MobileSpeechModel
import VGGBaseModel

class ModelSpeech(): # 语音模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出的拼音的表示大小是1422，即1421个拼音+1个空白块
        '''
        self.MS_OUTPUT_SIZE = 1422 #拼音类别是1421个，加上一个空白块，将output_dim设置为1422即可
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1000
        self.AUDIO_FEATURE_LENGTH = config.AUDIO_FEATURE_LENGTH
        speechmodel = MobileSpeechModel.SpeechModel()
        #speechmodel = VGGBaseModel.SpeechModel()
        self.base_model, self._model = speechmodel.create_model(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, self.MS_OUTPUT_SIZE, self.label_max_string_length)

        self.datapath = datapath
        self.slash = ''
        system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
        if(system_type == 'Windows'):
            self.slash='\\' # 反斜杠
        elif(system_type == 'Linux'):
            self.slash='/' # 正斜杠
        else:
            print('*[Message] Unknown System\n')
            self.slash='/' # 正斜杠
        if(self.slash != self.datapath[-1]): # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

    def TrainModel(self, datapath, epoch = 2, save_step = 1000, batch_size = 32, filename = abspath + 'model_speech/m' + ModelName + '/speech_model'+ModelName):
        '''
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        '''
        data=DataSpeech(datapath, 'train')
        num_data = data.GetDataNum() # 获取数据的数量
        yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH)

        for epoch in range(epoch): # 迭代轮数
            print('[running] train epoch %d .' % epoch)
            n_step = 0 # 迭代数据数
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+'%(epoch, n_step*save_step))
                    # data_genetator是一个生成器函数
                    self._model.fit_generator(yielddatas, save_step)
                    n_step += 1
                except StopIteration:
                    print('[error] generator error. please check data format.')
                    break

                self.SaveModel(comment='_e_'+str(epoch)+'_step_'+str(n_step * save_step))
                self.TestModel(self.datapath, str_dataset='train', data_count = 4)
                self.TestModel(self.datapath, str_dataset='dev', data_count = 4)

    def LoadModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName+'.model'):
        '''
        加载模型参数
        '''
        self._model.load_weights(filename)
        self.base_model.load_weights(filename + '.base')

    def SaveModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName,comment=''):
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

    def TestModel(self, datapath='', str_dataset='dev', data_count = 32, out_report = False, show_ratio = True, io_step_print = 10, io_step_file = 10):
        '''
        测试检验模型效果

        io_step_print
            为了减少测试时标准输出的io开销，可以通过调整这个参数来实现

        io_step_file
            为了减少测试时文件读写的io开销，可以通过调整这个参数来实现

        '''
        data=DataSpeech(self.datapath, str_dataset)
        #data.LoadDataList(str_dataset)
        num_data = data.GetDataNum() # 获取数据的数量
        if(data_count <= 0 or data_count > num_data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data

        try:
            ran_num = random.randint(0,num_data - 1) # 获取一个随机数

            words_num = 0
            word_error_num = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
            if(out_report == True):
                txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8') # 打开文件并读入

            txt = '测试报告\n模型编号 ' + ModelName + '\n\n'
            for i in range(data_count):
                data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据

                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                num_bias = 0
                while(data_input.shape[0] > self.AUDIO_LENGTH):
                    print('*[Error]','wave data', data_input.shape, 'is too long.','\n A Exception raise when test Speech Model.')
                    num_bias += 1
                    data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)  # 从随机数开始连续向后取一定数量数据

                pre = self.Predict(data_input, data_input.shape[0] // 8)

                words_n = data_labels.shape[0] # 获取每个句子的字数
                words_num += words_n # 把句子的总字数加上
                edit_distance = GetEditDistance(data_labels, pre) # 获取编辑距离
                if(edit_distance <= words_n): # 当编辑距离小于等于句子字数时
                    word_error_num += edit_distance # 使用编辑距离作为错误字数
                else: # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_error_num += words_n # 就直接加句子本来的总字数就好了

                if((i % io_step_print == 0 or i == data_count - 1) and show_ratio == True):
                    #print('测试进度：',i,'/',data_count)
                    print('Test Count: ',i,'/',data_count)


                if(out_report == True):
                    if(i % io_step_file == 0 or i == data_count - 1):
                        txt_obj.write(txt)
                        txt = ''

                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'

            #print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
            print('*[Test Result] Speech Recognition ' + str_dataset + ' set word error ratio: ', word_error_num / words_num * 100, '%')
            if(out_report == True):
                txt += '*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率： ' + str(word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt = ''
                txt_obj.close()

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

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

        #print('base_pred:\n', base_pred)

        #y_p = base_pred
        #for j in range(200):
        #       mean = np.sum(y_p[0][j]) / y_p[0][j].shape[0]
        #       print('max y_p:',np.max(y_p[0][j]),'min y_p:',np.min(y_p[0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][j][100])
        #       print('argmin:',np.argmin(y_p[0][j]),'argmax:',np.argmax(y_p[0][j]))
        #       count=0
        #       for i in range(y_p[0][j].shape[0]):
        #           if(y_p[0][j][i] < mean):
        #               count += 1
        #       print('count:',count)

        base_pred =base_pred[:, :, :]
        #base_pred =base_pred[:, 2:, :]

        r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1)

        #print('r', r)


        r1 = K.get_value(r[0][0])
        #print('r1', r1)


        #r2 = K.get_value(r[1])
        #print(r2)

        r1=r1[0]

        return r1
        pass

    def RecognizeSpeech(self, wavsignal, fs):
        '''
        最终做语音识别用的函数，识别一个wav序列的语音
        不过这里现在还有bug
        '''

        #data = self.data
        #data = DataSpeech('E:\\语音数据集')
        #data.LoadDataList('dev')
        # 获取输入特征
        data_input = GetMfccFeature(wavsignal, fs, config.AUDIO_MFCC_FEATURE_LENGTH)
        #t0=time.time()
        #data_input = GetFrequencyFeature3(wavsignal, fs)
        #t1=time.time()
        #print('time cost:',t1-t0)

        input_length = len(data_input)
        input_length = input_length // 8

        data_input = np.array(data_input, dtype = np.float)
        #print(data_input,data_input.shape)
        data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
        #t2=time.time()
        r1 = self.Predict(data_input, input_length)
        #t3=time.time()
        #print('time cost:',t3-t2)
        list_symbol_dic = GetSymbolList(self.datapath) # 获取拼音列表


        r_str=[]
        for i in r1:
            r_str.append(list_symbol_dic[i])

        return r_str
        pass

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


if(__name__=='__main__'):

    #import tensorflow as tf
    #from keras.backend.tensorflow_backend import set_session
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #进行配置，使用70%的GPU
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.95
    #config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    #set_session(tf.Session(config=config))


    datapath =      abspath + ''
    modelpath =  abspath + 'model_speech'


    if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
        os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

    system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
    if(system_type == 'Windows'):
        datapath = 'E:\\语音数据集'
        modelpath = modelpath + '\\'
    elif(system_type == 'Linux'):
        datapath =      abspath + 'dataset'
        modelpath = modelpath + '/'
    else:
        print('*[Message] Unknown System\n')
        datapath = 'dataset'
        modelpath = modelpath + '/'

    ms = ModelSpeech(datapath)


    #ms.LoadModel(modelpath + 'm251/speech_model251_e_0_step_100000.model')
    ms.TrainModel(datapath, epoch = 50, batch_size = 16, save_step = 500)

    #t1=time.time()
    #ms.TestModel(datapath, str_dataset='train', data_count = 128, out_report = True)
    #ms.TestModel(datapath, str_dataset='dev', data_count = 128, out_report = True)
    #ms.TestModel(datapath, str_dataset='test', data_count = 128, out_report = True)
    #t2=time.time()
    #print('Test Model Time Cost:',t2-t1,'s')
    #r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00241I0053.wav')
    #r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00020I0087.wav')
    #r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\train\\A11\\A11_167.WAV')
    #r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\test\\D4\\D4_750.wav')
    #print('*[提示] 语音识别结果：\n',r)
