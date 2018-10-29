import sys
import tensorflow as tf

import numpy as np
from general_function.file_wav import read_wav_data, GetFrequencyFeature3
from keras import backend as K
# from general_function.file_dict import *
# from general_function.gen_func import GetFrequencyFeature3

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200

filename = sys.argv[1]

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = 'model_speech/speech_model251_e_0_step_500.base.h5.pb'

    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    wavsignal,fs = read_wav_data(filename)
    data_input = GetFrequencyFeature3(wavsignal, fs)
    input_length = len(data_input)
    input_length = input_length // 8

    data_input = np.array(data_input, dtype = np.float)
    print('data input', data_input, data_input.shape)
    data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    batch_size = 1
    in_len = np.zeros((batch_size),dtype = np.int32)
    in_len[0] = input_length
    x_in = np.zeros((batch_size, 1600, AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
    for i in range(batch_size):
        x_in[i,0:len(data_input)] = data_input
    print('x_in', x_in)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        # tf.initialize_all_variables().run()
        input_x = sess.graph.get_tensor_by_name("the_input:0")
        print(input_x)
        output = sess.graph.get_tensor_by_name("output_node0:0")
        print(output)

        res1 = sess.run(output, {input_x:x_in})

        print('res1', type(res1), res1)
        res11 =res1[:,:,:]

        r = K.ctc_decode(res11, in_len, greedy = True, beam_width=100, top_paths=1)
        print('r:', r)
        r1 = K.get_value(r[0][0])
        print('r1:', type(r1), r1)
        r1=r1[0]
        print('r1:', type(r1), r1)
