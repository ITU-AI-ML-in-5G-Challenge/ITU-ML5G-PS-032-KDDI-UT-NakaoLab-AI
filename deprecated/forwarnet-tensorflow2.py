import itertools
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取csv
dataset = pd.read_csv('./csv/physical-20200629.csv')

print('列数:', dataset.shape[1], '行数:', dataset.shape[0])
# 划分训练测试
column = dataset.columns
X_all = dataset[column[:-2]]
y_all = dataset[column[-1]]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
ss = StandardScaler()
std_X_train = ss.fit_transform(X_train)
std_X_test = ss.fit_transform(X_test)

class Model(object):
    def __init__(self, input_size):
        self.W1 = tf.Variable(tf.random.normal([input_size, 100]))  # 随机初始化张量参数
        self.b1 = tf.Variable(tf.random.normal([100]))

        self.W2 = tf.Variable(tf.random.normal([100, 13]))
        self.b2 = tf.Variable(tf.random.normal([13]))

    def __call__(self, x):
        x = tf.cast(x, tf.float32)  # 转换输入数据类型
        # 线性计算 + RELU 激活
        fc1 = tf.nn.relu(tf.add(tf.matmul(x, self.W1), self.b1))
        fc2 = tf.add(tf.matmul(fc1, self.W2), self.b2)
        return fc2

def loss_fn(model, x, y):
    preds = model(x)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y))

def accuracy_fn(logits, labels):
    preds = tf.argmax(logits, axis=1)  # 取值最大的索引，正好对应字符标签
    labels = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

EPOCHS = 100  # 迭代此时
LEARNING_RATE = 0.02  # 学习率
model = Model(X_train.shape[1])  # 模型
for epoch in range(EPOCHS):
    with tf.GradientTape() as tape:  # 追踪梯度
        loss = loss_fn(model, X_train, y_train)

    trainable_variables = [model.W1, model.b1, model.W2, model.b2]  # 需优化参数列表
    grads = tape.gradient(loss, trainable_variables)  # 计算梯度

    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)  # 优化器
    optimizer.apply_gradients(zip(grads, trainable_variables))  # 更新梯度
    accuracy = accuracy_fn(model(X_test), y_test)  # 计算准确度

    # 输出各项指标
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Train loss: {:.3f}, Test accuracy: {:.3f}'
              .format(epoch+1, EPOCHS, loss, accuracy))