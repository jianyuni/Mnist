import numpy   #导入数据库
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import tensorflow.keras
import requests
requests.packages.urllib3.disable_warnings
import ssl
import matplotlib.pyplot as plt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
 
seed = 7   #设置随机种子
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data() #加载数据
'''
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
'''
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
#数据集是3维的向量（instance length,width,height).对于多层感知机，模型的输入是二维的向量，因此这
#里需要将数据集reshape，即将28*28的向量转成784长度的数组。可以用numpy的reshape函数轻松实现这个过
#程。
 
#给定的像素的灰度值在0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1。
X_train = X_train / 255
X_test = X_test / 255
 
#最后，模型的输出是对每个类别的打分预测，对于分类结果从0-9的每个类别都有一个预测分值，表示将模型
#输入预测为该类的概率大小，概率越大可信度越高。由于原始的数据标签是0-9的整数值，通常将其表示成#0ne-hot向量。如第一个训练数据的标签为5，one-hot表示为[0,0,0,0,0,1,0,0,0,0]。
 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
 
#现在需要做得就是搭建神经网络模型了，创建一个函数，建立含有一个隐层的神经网络。
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

 
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
