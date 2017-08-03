#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:09:40 2017

@author: johnnyhsieh
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv'
                       ,engine = 'python')
dataset.head()
#remove the nan data if developer row is nan
dataset= dataset.dropna(subset = ['Platform','Publisher','Genre'
                                            ,'Developer','Rating'],how='any')
y_train = dataset['Global_Sales']
x_train = dataset.drop(labels = ['Name','Year_of_Release','Global_Sales'
                                 ,'User_Score','User_Count'
                                 ,'Critic_Score','Critic_Count'], axis = 1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x_train['Platform'] = le.fit_transform(x_train['Platform'])
x_train['Publisher'] = le.fit_transform(x_train['Publisher'].astype(str))
x_train['Genre'] = le.fit_transform(x_train['Genre'])
x_train['Developer'] = le.fit_transform(x_train['Developer'])
x_train['Rating'] = le.fit_transform(x_train['Rating'].astype(str))

ohe = OneHotEncoder(categorical_features = [0,1,2,7])
x_train = ohe.fit_transform(x_train).toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train
                                                 ,test_size = 0.2
                                                 ,random_state = 42)


import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping


Regression = Sequential()
Regression.add(Dense(units = 2048,activation= 'linear',kernel_initializer='uniform'
                     ,input_dim = 2035))
Regression.add(Dense(units = 1024, activation = 'linear',kernel_initializer='uniform'))
Regression.add(Dense(units = 512, activation = 'sigmoid',kernel_initializer='uniform'))
Regression.add(Dense(units = 128, activation = 'sigmoid',kernel_initializer='uniform'))
Regression.add(Dense(units = 64, activation = 'sigmoid',kernel_initializer='uniform'))
Regression.add(Dense(units = 32, activation = 'sigmoid',kernel_initializer='uniform'))
Regression.add(Dense(units = 1,activation = 'linear',kernel_initializer='uniform'))
Regression.compile(optimizer = 'adam',loss = 'mse')
#early_stop = EarlyStopping(monitor = 'val_loss',patience = 10,verbose = 0,mode = 'auto')
#tbCallBack = TensorBoard(log_dir='/tmp/keras_logs'
                                         #, write_graph=True
                                         #, write_images = True)

Regression.fit(x_train,y_train,batch_size = 32,epochs = 30
               ,validation_data = (x_test,y_test))

score = Regression.evaluate(x_test, y_test, batch_size=32)
# Predicting the Test set results
y_pred = Regression.predict(x_train)
# calculate the accuracy of our model
acc = [sum(y_pred)/sum(np.float32(y_train))]

#see hows our model works
import matplotlib.pyplot as plt
plt.plot(np.float32(y_train), color ='red', label = 'real sale')
plt.plot(y_pred, color = 'blue', label = 'predict sale')
plt.title('video game predict')
plt.ylabel('video game sale')
plt.legend
plt.show()

reslut_Y = pd.DataFrame(y_train)

