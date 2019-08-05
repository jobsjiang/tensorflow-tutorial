# -*- coding: utf-8 -*-
# Convolutional AutoEncoder for denoising Cifar dataset
# Backend tensorflow and Import

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np

# 设置基本参数
batch_size = 32
num_classes = 10
epochs = 100
saveDir = "./opt/files1/python/transfer/ae/"   # 训练的参数文件保存路径
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
# 加载 Cifar10 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# 把 x_test 分成验证集（validation）和测试集（test）
x_val = x_test[:7000]
x_test = x_test[7000:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print("validation data: {0} \ntest data: {1}".format(x_val.shape, x_test.shape))


# 给数据添加噪声
# 噪声因子控制图像的噪声
noise_factor = 0.1 
# 引入高斯随机噪声
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_val_noisy = x_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_val.shape)
# 区间剪切，超过区间会被转成区间极值，以确保表示图像的特征向量的元素在0和1之间
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
x_val_noisy = np.clip(x_val_noisy, 0., 1.)



# 在原图和加噪声图中各选取十张绘图显示比对
def showOrigDec(orig, noise, num=10):
    import matplotlib.pyplot as plt  # 可视化
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(orig[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(noise[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
# 挑取数据集中部分图片显示
showOrigDec(x_train, x_train_noisy)
showOrigDec(x_train[100:], x_train_noisy[100:])
showOrigDec(x_train[200:], x_train_noisy[200:])



# 去噪卷积自编码器
input_img = Input(shape=(32, 32, 3))  #input层   (？,32, 32, 3)  ？：可变随机样本数
x = Conv2D(32, (3, 3), padding='same')(input_img)   #卷积层  (？,32, 32, 32)
x = BatchNormalization()(x)     # BN层   对于每个神经元做归一化处理
x = Activation('relu')(x)   # 加入激活函数'ReLu', 只保留大于0 的值
x = MaxPooling2D((2, 2), padding='same')(x)     #最大池化层    (？,16, 16, 32)
x = Conv2D(32, (3, 3), padding='same')(x)     # (？,16, 16, 32)
x = BatchNormalization()(x)     
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)   # 编码器    (？,8, 8, 32)

x = Conv2D(32, (3, 3), padding='same')(encoded)     #(？,8, 8, 32)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)  # 上采样层  (？,16, 16, 32)
x = Conv2D(32, (3, 3), padding='same')(x)   # (？,16, 16, 32)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)   # (？,32, 32, 32)
x = Conv2D(3, (3, 3), padding='same')(x)   # (？,32, 32, 3)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)   # 解码器  加入激活函数'sigmoid'

model = Model(input_img, decoded)   #选定模型的输入，decoded（即输出）的格式
model.compile(optimizer='adam', loss='binary_crossentropy') #定义优化目标和损失函数   优化器 Adam  loss： 交叉熵损失

# 训练自编码器 
# load pretrained weights
# model.load_weights(saveDir + "AutoEncoder_Cifar10_denoise_weights.10-0.55-0.55.hdf5")
# 当被监测的数量不在提升的时候停止训练
es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
# 模型参数保存
chkpt = saveDir + 'AutoEncoder_Cifar10_denoise_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
# 在每个epoch后保存模型到filepath
cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
# 训练
history = model.fit(x_train_noisy, x_train,  # 输入输出
                    batch_size=batch_size,
                    epochs=epochs,  # 迭代次数
                    verbose=1,
                    validation_data=(x_val_noisy, x_val),
                    callbacks=[es_cb, cp_cb],
                    shuffle=True)

# Evaluate with test dataset
score = model.evaluate(x_test_noisy, x_test, verbose=1)
print(score)


# Visualize original image, noisy image and denoised image
c10test = model.predict(x_test_noisy)  #测试集 输入去噪网络之后输出去噪结果。
c10val = model.predict(x_val_noisy)     #验证集 输入去噪网络之后输出去噪结果。
print("Cifar10_test: {0}\nCifar10_val: {1}".format(np.average(c10test), np.average(c10val)))


# 在测试集、验证集 中选 原图、加噪声图、去噪图中 各选取十张绘图显示比对
def showOrigDec(orig, noise, denoise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(orig[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noise[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display denoised image
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(denoise[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
showOrigDec(x_test, x_test_noisy, c10test)
showOrigDec(x_test[100:], x_test_noisy[100:], c10test[100:])
showOrigDec(x_test[200:], x_test_noisy[200:], c10test[200:])
showOrigDec(x_val, x_val_noisy, c10val)
showOrigDec(x_val[100:], x_val_noisy[100:], c10val[100:])
showOrigDec(x_val[200:], x_val_noisy[200:], c10val[200:])



# Try the Denoising AutoEncoder on Cifar100
# Load cifar100 dataset
from keras.datasets import cifar100
(x_train100, y_train100), (x_test100, y_test100) = cifar100.load_data()

# normalize data
x_train100 = x_train100.astype('float32')
x_test100 = x_test100.astype('float32')
x_train100 /= 255
x_test100 /= 255

print('x_train100 shape:', x_train100.shape)
print(x_train100.shape[0], 'train samples')
print(x_test100.shape[0], 'test samples')

## add noise to Cifar100 data
noise_factor = 0.1
x_train100_noisy = x_train100 + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train100.shape)
x_test100_noisy = x_test100 + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test100.shape)

x_train100_noisy = np.clip(x_train100_noisy, 0., 1.)
x_test100_noisy = np.clip(x_test100_noisy, 0., 1.)

score = model.evaluate(x_train100_noisy, x_train100_noisy, verbose=1)
print(score)

score = model.evaluate(x_test100_noisy, x_test100_noisy, verbose=1)
print(score)

# Autoencoder on  Cifar100 dateset
c100train = model.predict(x_train100_noisy)
c100test = model.predict(x_test100_noisy)

print("Cifar100 train: {0} \nCifar100 test: {1}".format(np.average(c100train), np.average(c100test)))

showOrigDec(x_train100, x_train100_noisy, c100train)
showOrigDec(x_train100[100:], x_train100_noisy[100:], c100train[100:])
showOrigDec(x_train100[200:], x_train100_noisy[200:], c100train[200:])
showOrigDec(x_test100, x_test100_noisy, c100test)
showOrigDec(x_test100[100:], x_test100_noisy[100:], c100test[100:])
showOrigDec(x_test100[200:], x_test100_noisy[200:], c100test[200:])
