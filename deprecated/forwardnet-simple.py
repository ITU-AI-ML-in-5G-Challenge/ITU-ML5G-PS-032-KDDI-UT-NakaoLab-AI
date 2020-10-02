#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import sys
import struct,socket
import operator
import os
from datetime import datetime, timedelta
import time

if 1:
    TrainUEDataPath='../UE/44003_train/';
    TestUEDataPath='../UE/44003_test/';
else:    
    TrainUEDataPath='../UE/44190_train/';
    TestUEDataPath='../UE/44190_test/';

POPAPP= 2;
DIM = 2;
LAYERS = 5;
EPOCH = 20000;
LEARNING_RATE = 1e-4;


def SPEC(x):
    if (x<23) and (x>21):
        result=[1, 0];
    else:
        result=[0, 1];
    return result;

def toseconds(xt):
    sec=xt.seconds;
    microsec=xt.microseconds;
    return sec+1.0*microsec/1e6;


def read_data(filename):
    flog = open(filename, 'r');
    x_vec = [];
    y_vec = [];
    tIn=datetime.now();
    tOut=datetime.now();
    exio=0;
    exoi=0;
    print(filename);
    k=0;
    exevent="start"
    for line in flog:
        a = line.strip().split(',');
        k=k+1;
        if (k==1):
            continue;
        
        event = a[2];
        str_time=a[3];
        
        dt = datetime.strptime(str_time,'%Y-%m-%d %H:%M:%S.%f+09:00');
        #print dt
        if (event=="CheckIn") and (exevent=="CheckOut"):
            tIn=dt;
            coi=tIn-tOut;
            xt=[exoi, exio];
            yt=SPEC(toseconds(coi));
            x_vec.append(xt);
            y_vec.append(yt);
            
            exoi=min(toseconds(coi), 5000);
        
        if (event=="CheckOut") and (exevent=="CheckIn"):
            tOut=dt;
            cio=tOut-tIn;
            exio=min(toseconds(cio), 5000);
        exevent=event;
            
    flog.close();
    vec_size = len(y_vec);
    return x_vec, y_vec, vec_size

# %%
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# %%
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def add_layer(x_in, in_size, out_size):
    W_h = weight_variable([in_size, out_size]);
    b_h = bias_variable([out_size]);
    y_h = tf.matmul(x_in, W_h) + b_h;
    return y_h
    

LAYER_SIZE = 100

# %%

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# %%
## Create the model

    
x_initializer = tf.placeholder(dtype=tf.float32, shape=[None, DIM])
y_initializer = tf.placeholder(dtype=tf.float32, shape=[None, POPAPP])
    
x = tf.Variable(x_initializer, trainable=False, validate_shape=False, collections=[])
y_ = tf.Variable(y_initializer, trainable=False, validate_shape=False, collections=[])


L1 = add_layer(x, DIM, LAYER_SIZE);
LH = tf.nn.relu(L1);

for _ in range (LAYERS):
    LH = add_layer(LH, LAYER_SIZE, LAYER_SIZE);
    LH = tf.nn.relu(LH);

y = add_layer(LH, LAYER_SIZE, POPAPP);

saver = tf.train.Saver()


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
training = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy);


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

i = 0;

var_x_train = locals()
var_y_train = locals()

x_train=[];
y_train=[];
for subdir, dirs, files in os.walk(TrainUEDataPath):
    for f in files:
        trainfile = TrainUEDataPath + f;
        i = i + 1;
        var_x_train['x_train_'+str(i)], var_y_train['y_train_'+str(i)], train_size = read_data(trainfile)
        print (trainfile, train_size)
        x_train.extend(locals()['x_train_'+str(i)]);
        y_train.extend(locals()['y_train_'+str(i)]);

TRAIN_NUM = i;


#validatefile = 'validatevec/app-2019-09-%02d.vec' % (15)
#x_validate, y_validate, validate_size = read_tim_data(validatefile)

init = tf.global_variables_initializer();

sess.run(init)



# %%
sess.run(x.initializer, feed_dict={x_initializer: x_train})
sess.run(y_.initializer, feed_dict={y_initializer: y_train}) 

# %%
model_path = "dnn/fwdnet"+".ckpt"
ckptflag = "%s.index" %(model_path)
if os.path.exists(ckptflag):
    load_path = saver.restore(sess, model_path);


last_train_accuracy = 0.0;
max_train_accuracy = 0.0;
decline = 1.0;


# %%
import time

for i in range(EPOCH):
    sess.run([training])
    if i%100 == 0:
        print('Training Step: ' +str(i) + '   [cross_entropy, accuracy] = ' + str(sess.run([cross_entropy, accuracy], feed_dict={x:x_train, y_:y_train})))

        save_path = saver.save(sess, model_path);
        train_accuracy = float(sess.run(accuracy))
        if (train_accuracy > 0.98):
            break;

x_total=[];
y_total=[];
print ('#########Testing############')

for subdir, dirs, files in os.walk(TestUEDataPath):
    for f in files:
        testfile = TestUEDataPath + f;
        if not os.path.exists(testfile):
            continue;
        x_test, y_test, test_size = read_data(testfile)
        print (testfile, test_size)
        x_total.extend(x_test);
        y_total.extend(y_test);
        print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

print("Avg rate = "+ str(sess.run(accuracy, feed_dict={x: x_total, y_: y_total})))

