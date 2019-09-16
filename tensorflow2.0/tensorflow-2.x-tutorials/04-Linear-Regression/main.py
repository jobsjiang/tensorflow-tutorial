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