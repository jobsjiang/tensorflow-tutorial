# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,datasets
def prepare_mnist_features_labels(x,y):
    x = tf.cast(x,tf.float32) / 255.0
    y = tf.cast(y,tf.int64)
    return x,y