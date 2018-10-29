#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于训练语音识别系统语音模型的程序

"""
import os
import tensorflow as tf

from speech_model_251 import ModelSpeech

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#进行配置，使用95%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
tf.keras.backend.set_session(tf.Session(config=config))

datapath = 'dataset/'
modelpath = 'model_speech'
ms = ModelSpeech(datapath)

ms.TrainModel(datapath, epoch = 50, batch_size = 16, save_step = 500)
