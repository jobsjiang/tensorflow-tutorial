# -*- coding：utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
class Regressor(keras.layers.Layer):
    def __init__(self):
        super(Regressor,self).__init__()
        # [dim_in,dim_out]
        self.w = self.add_variable('meanless-name',[13,1])
        # [dim_out]
        self.b = self.add_variable('meanless-name',[1])

        print(self.w.shape,self.b.shape)
        print(type(self.w),tf.is_tensor(self.w),self.w.name)
        print(type(self.b),tf.is_tensor(self.b),self.b.name)

    def call(self,x):
        x = tf.matmul(x,self.w) + self.b
        return x

def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
