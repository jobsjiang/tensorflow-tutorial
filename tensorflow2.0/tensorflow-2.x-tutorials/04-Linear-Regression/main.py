# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

class Regressor(keras.layers.Layer):
    def __init__(self):
        super(Regressor,self).__init__()