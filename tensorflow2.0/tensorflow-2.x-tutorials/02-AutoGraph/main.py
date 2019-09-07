# -*- coding：utf-8 -*-
import tensorflow as tf
import timeit

cell = tf.keras.layers.LSTMCell(10)

@tf.function
def fn(input,state):
    """
    利用静态图计算LSTM
    :param input:
    :param state:
    :return:
    """
    return cell(input,state)
