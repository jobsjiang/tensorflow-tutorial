# -*- coding：utf-8 -*-
import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
(xs,ys,),_ = datasets.mnist.load_data()
print('datasets:',xs.shape,ys.shape,xs.min(),xs.max())
xs = tf.convert_to_tensor(xs,dtype=tf.float32) / 255

