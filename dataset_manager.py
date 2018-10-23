'''
管理dataset
'''

import numpy as np
from general_function.file_wav import read_wav_data, GetMfccFeature
import config

class DataSetManager(object):

    def __init__(self, data_path='./dataset/'):
        self.data_path = data_path
        self.list_symbol = self.get_symbol_list() # 全部汉语拼音符号列表
        self.data_gen = self.data_generator_()

    def data_generator(self, batch_size=32, audio_length = 1600):
        '''
        数据生成器函数
        batch_size: 一次产生的数据量
        需要再修改。。。
        '''
        labels = []
        for i in range(0,batch_size):
            labels.append([0.0])
        labels = np.array(labels, dtype = np.float)

        while True:
            X = np.zeros((batch_size, audio_length, config.AUDIO_FEATURE_LENGTH, 1), dtype = np.float)
            y = np.zeros((batch_size, 64), dtype=np.int16)
            input_length = []
            label_length = []
            i = 0
            while i < batch_size:
                data_input, data_labels = next(self.data_gen)
                if data_input.shape[0] * 1.05 > audio_length:
                    #print('ignore:', data_input.shape)
                    continue

                input_length.append(data_input.shape[0] // 8 + data_input.shape[0] % 8)
                X[i,0:len(data_input)] = data_input
                y[i,0:len(data_labels)] = data_labels
                label_length.append([len(data_labels)])
                i+=1

            label_length = np.matrix(label_length)
            input_length = np.array(input_length).T
            yield [X, y, input_length, label_length ], labels

    def data_generator_(self):
        while True:
            wav_index, syllable_index = self.load_thchs30()
            for x in self.dataset_generator(wav_index, syllable_index):
                yield x

            wav_index, syllable_index = self.load_stcmd()
            for x in self.dataset_generator(wav_index, syllable_index):
                yield x

            wav_index, syllable_index = self.load_primewords()
            for x in self.dataset_generator(wav_index, syllable_index):
                yield x

    def dataset_generator(self, wav_index, syllable_index):
        for tag in wav_index:
            wavfp = wav_index[tag]
            syllable = syllable_index[tag]

            # TODO: debug
            data_input = np.random.random((300, config.AUDIO_FEATURE_LENGTH))
            print(wavfp)
            #wavsignal, fs = read_wav_data(wavfp)
            #data_input = GetMfccFeature(wavsignal, fs, numcep=config.AUDIO_MFCC_FEATURE_LENGTH)
            #print(wavfp)
            data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
            # 获取输入特征
            syllable_num = list(map(self.get_symbol_num, syllable))
            yield data_input, np.array(syllable_num)

    def read_data(self):
        pass

    def load_thchs30(self):
        '''
        '''
        train_wav_lst = self.data_path + 'thchs30/' + 'train.wav.lst'
        train_syllable_lst = self.data_path + 'thchs30/' + 'train.syllable.txt'
        wav_index = {}
        syllable_index = {}

        with open(train_wav_lst, 'r') as r:
            for line in r.readlines():
                if line.strip() == '':
                    continue
                x, y = line.strip().split(' ')
                wav_index[x] = y

        with open(train_syllable_lst, 'r') as r:
            for line in r.readlines():
                if line.strip() == '':
                    continue
                linesp = line.strip().split(' ')
                x = linesp[0]
                y = linesp[1:]
                syllable_index[x] = y

        print('load thchs30')
        print(len(wav_index))
        print(len(syllable_index))
        return wav_index, syllable_index

    def load_stcmd(self):
        train_wav_lst = self.data_path + 'st-cmds/' + 'train.wav.txt'
        train_syllable_lst = self.data_path + 'st-cmds/' + 'train.syllable.txt'
        wav_index = {}
        syllable_index = {}

        with open(train_wav_lst, 'r') as r:
            for line in r.readlines():
                if line.strip() == '':
                    continue
                x, y = line.strip().split(' ')
                wav_index[x] = y

        with open(train_syllable_lst, 'r') as r:
            for line in r.readlines():
                if line.strip() == '':
                    continue
                linesp = line.strip().split(' ')
                x = linesp[0]
                y = linesp[1:]
                syllable_index[x] = y

        print('load stcmd')
        print(len(wav_index))
        print(len(syllable_index))
        return wav_index, syllable_index

    def load_primewords(self):
        train_wav_lst = self.data_path + 'primewords/' + 'train.wav.txt'
        wav_index = {}
        syllable_index = {}
        tag = 1
        with open(train_wav_lst, 'r') as r:
            for line in r.readlines():
                if line.strip() == '':
                    continue
                x, _, y = line.strip().split('###')
                wav_index[tag] = y
                syllable_index[tag] = x.split(' ')
                tag += 1

        print('load primewords')
        print(len(wav_index))
        print(len(syllable_index))
        return wav_index, syllable_index

    def get_symbol_list(self):
        '''
        加载拼音符号列表，用于标记符号
        返回一个列表list类型变量
        '''
        txt_obj = open('dict.txt','r',encoding='UTF-8') # 打开文件并读入
        txt_text = txt_obj.read()
        txt_lines = txt_text.split('\n') # 文本分割
        list_symbol = [] # 初始化符号列表
        for i in txt_lines:
            if i != '' :
                txt_l = i.split('\t')
                list_symbol.append(txt_l[0])
        txt_obj.close()
        list_symbol.append('_')
        self.symbol_num = len(list_symbol)
        return list_symbol

    def get_symbol_num(self, symbol):
        if symbol != '':
            return self.list_symbol.index(symbol)
        return self.symbol_num

if __name__ == '__main__':
    manager = DataSetManager()
    # manager.load_stcmd()
    #manager.load_primewords()
    #wav_index, syllable_index =  manager.load_thchs30()
    #for x in manager.dataset_generator(wav_index, syllable_index):
    #     print(x)

    # test
    data_gen = manager.data_generator()
    print(len(next(data_gen)[1]))
    #print(next(data_gen)[1])
