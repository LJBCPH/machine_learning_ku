# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:56:21 2021

@author: lucas
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from libsvm.svmutil import * #https://www.csie.ntu.edu.tw/~cjlin/libsvm/
from sklearn import svm #Sklearn
import matplotlib.pyplot as plt

y_train = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H04/data/y_train1-1.csv', delimiter =',')
y_test = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H04/data/y_test1-1.csv', delimiter =',')
x_train = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H04/data/X_train_binary.csv', delimiter =',')
x_test = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H04/data/X_test_binary.csv', delimiter =',')


#====================================# 1.1
freq_train = pd.DataFrame([len(y_train[y_train == -1])/len(y_train), len(y_train[y_train == 1])/len(y_train)])
print(freq_train.to_latex())
freq_test = pd.DataFrame([len(y_test[y_test == -1])/len(y_test), len(y_test[y_test== 1])/len(y_test)])
print(freq_test.to_latex())

def f_norm(x, mean_vec, variance_vec):
    x = (x-mean_vec) / variance_vec
    return x
    
x_var = np.std(x_train, axis = 0)
x_mean = np.mean(x_train, axis = 0)

x_train_trans = f_norm(x_train, x_mean, x_var)
x_test_trans = f_norm(x_test, x_mean, x_var)

std_mean_test = pd.DataFrame([x_test_trans.std(axis = 0)**2, x_test_trans.mean(axis = 0)])
print(std_mean_test.to_latex())

x_test_trans.std(axis = 0)**2
std_mean_test

print(pd.DataFrame(np.array([x_test_trans.std(axis = 0)**2, x_test_trans.mean(axis = 0)])).T.to_latex())

#====================================# 1.2
x_trans_split = np.array_split(x_train_trans, 5)
y_train_split = np.array_split(y_train, 5)

x_train_trans[0:30,:]
gamma = np.array([0.001, 0.01, 0.1, 1, 10, 20]) * 0.1
C = np.array([0.01, 0.1, 1, 2.5, 5, 10, 25])

validation_err = np.ones([len(C),len(gamma)])
n_splits = 5
x_train_split = np.array_split(x_train_trans, n_splits)
y_train_split = np.array_split(y_train, n_splits)
for i in range(len(C)):
    for j in range(len(gamma)):
        err_cv = 0
        for k in range(5):
            x_train_split, x_test_split = np.concatenate((x_train_trans[:(k*30), :], x_train_trans[((k+1)*30):, :])), x_train_trans[(k*30):((k+1)*30), :]
            y_train_split, y_test_split = np.concatenate((y_train[:(k*30)], y_train[((k+1)*30):])), y_train[(k*30):((k+1)*30)]
            model = svm.SVC(C=C[i],kernel='rbf',gamma=gamma[j],max_iter=1000000)
            model.fit(x_train_split, y_train_split)
            preds = model.predict(x_test_split)
            err = np.sum(preds == y_test_split) / (len(x_train_trans) / n_splits)
            err_cv += 1-err
        validation_err[i,j] = err_cv / n_splits

print(pd.DataFrame(validation_err, columns = gamma, index=C).to_latex())

model = svm.SVC(C=500,kernel='rbf',gamma=0.001,max_iter=1000000)
model.fit(x_train_trans, y_train)

preds = model.predict(x_test_trans)
err = np.sum(preds == y_test) / len(x_train_trans)

#=============================================# 1.3
C_vec = np.linspace(1e-08, 400, 100)
sup_vec_1 = np.ones([len(C_vec)])
for count, i in enumerate(C_vec):
    model = svm.SVC(C=i,kernel='rbf',gamma=0.001,max_iter=1000000)
    model.fit(x_train_trans, y_train)
    sup_vec_1[count] = model.n_support_[0] + model.n_support_[1]


fig, ax = plt.subplots(figsize = (6, 6))
plt.plot(C_vec, sup_vec_1, label = "Total bounded support vectors")
plt.legend(loc="upper right", fontsize=10)
plt.xlabel("C")
plt.ylabel("No. of bounded support vectors")
fig.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H04/plots/h04_1_3.png')
plt.show()

    
#=================================================# FLIGHT 2.1
res = sum(np.random.binomial(100, 0.95, 10000000) == 100)/10000000
hoeffding = np.exp(-2*100*0.05**2)
hoeffding
actual_prob = 0.95**100
actual_prob * 100

