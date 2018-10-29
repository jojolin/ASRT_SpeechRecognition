# A Deep-Learning-Based Chinese Speech Recognition System
基于深度学习的中文语音识别系统
[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) 
tensorflow == 1.11.0
python == 3.6

!! this is a working on project.

## Introduction 简介

基于深度学习的中文语音识别（目前只识别拼音序列）。

为了能在嵌入式版上面运行，添加了手机模型(基于mobilenet修改)和vgg模型。

通过tf.keras和SeparableConv2D提高性能。

训练基于三个数据集
[data\_thchs30.tgz](http://cn-mirror.openslr.org/18/)
[ST-CMDS-20170001\_1-OS.tar.gz](http://cn-mirror.openslr.org/18/)
[primewords\_md\_2018\_set1](http://cn-mirror.openslr.org/18/)

训练准确率有待提升

## 训练及使用
```shell
# 训练
$ python train_mspeech.py

# 使用
$ python speechcli.py model_speech/m251_pick/speech_model251_e_0_step_42000.model ../voices/*_3.wav
```

## 鸣谢
开源数据集提供方
[ASRT\_SpeechRecognition项目](https://github.com/nl8590687/ASRT_SpeechRecognition)

