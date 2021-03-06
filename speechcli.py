#!/usr/bin/env python3
"""
@author: nl8590687
用于测试语音识别系统语音模型的程序

"""
import time
import sys

from speech_model_251 import ModelSpeech

model = sys.argv[1]
sound_fp = sys.argv[2:]

datapath = 'data/'
modelpath = 'model_speech/'
ms = ModelSpeech(datapath)
ms.LoadModel(model)

pinyin = ms.RecognizeSpeech_FromFile(*sound_fp)
print('*[提示] 语音识别结果：\n',pinyin)
time.sleep(3)
#r = ml.SpeechToText(pinyin)
#r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00020I0087.wav')
#r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\train\\A11\\A11_167.WAV')
#r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\test\\D4\\D4_750.wav')
#print('*[提示] 语音识别结果：\n',r)


