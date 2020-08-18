#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
import struct,socket
import operator
import os
from datetime import datetime, timedelta
import time

if 0:
    TrainUEDataPath='../UE/44003_train/';
    TestUEDataPath='../UE/44003_test/';
else:    
    TrainUEDataPath='../UE/44190_train/';
    TestUEDataPath='../UE/44190_test/';



POPAPP= 2;
DIM = 2;
LAYERS = 5;
EPOCH = 200000;
LEARNING_RATE = 1e-4;


def SPEC(x):
    if (x<23) and (x>21):
        result=1;
    else:
        result=0;
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


var_x_train = locals()
var_y_train = locals()

lx_train=[];
ly_train=[];
i = 0;
for subdir, dirs, files in os.walk(TrainUEDataPath):
    for f in files:
        trainfile = TrainUEDataPath + f;
        i = i + 1;
        var_x_train['x_train_'+str(i)], var_y_train['y_train_'+str(i)], train_size = read_data(trainfile)
        #print (trainfile, train_size)
        lx_train.extend(locals()['x_train_'+str(i)]);
        ly_train.extend(locals()['y_train_'+str(i)]);


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def create_dataset(X, y):
    Xs, ys = [], []
    for i in range(len(y)):
        v = [[X[i][0]], [X[i][1]]]
        #v = [[X[i][0]]]
        Xs.append(v)        
        ys.append([y[i]])
    return np.array(Xs), np.array(ys)


X_train, y_train = create_dataset(lx_train, ly_train)
#X_test, y_test = create_dataset(test, test.sine)

print(X_train.shape)
print(y_train.shape)

model = keras.Sequential()
model.add(keras.layers.LSTM(
  units=128,
  input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dense(units=1))
model.compile(
  loss='mean_squared_error',
  optimizer=keras.optimizers.Adam(0.001)
)

history = model.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=16, 
    validation_split=0.1, 
    verbose=1, 
    shuffle=False
)
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend();
#plt.show();


print ('#########Testing############')

def calrate(ypred,y):
    #print ypred.values
    #print y
    k=0;
    for i in range(len(y)):
        if ypred[i][0]>0.5:
            m = 1;
        else:
            m= 0;
        if m == y[i]:
            k=k+1;
    return 1.0*k/len(y);
            

lx_total=[];
ly_total=[];
for subdir, dirs, files in os.walk(TestUEDataPath):
    for f in files:
        testfile = TestUEDataPath + f;
        if not os.path.exists(testfile):
            continue;
        lx_test, ly_test, test_size = read_data(testfile)
        print (testfile, test_size)
        lx_total.extend(lx_test);
        ly_total.extend(ly_test);

        X_test, y_test = create_dataset(lx_test, ly_test)
        y_pred=model.predict(X_test)
        print calrate(y_pred, ly_test);
        

X_test, y_test = create_dataset(lx_total, ly_total)
y_pred = model.predict(X_test)
print calrate(y_pred, y_test);

