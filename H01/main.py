# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

n = 20
sim = int(1e6)

## Flipping 1e6 times 20 coins
def generate_bernoulli(prob, n = n, sim = sim):
    sims = ([])
    for _ in range(sim):
        sims.append(np.random.binomial(n=1, p = prob, size=n).mean())
    ## Dumb way of getting the number of occurences
    frequencies = ([])
    for i in range(sim):
        if prob < 0.4:                
            if sims[i] >= 0.1:
                frequencies.append(0.1)
            if sims[i] >= 0.15:
                frequencies.append(0.15)
            if sims[i] >= 0.2:
                frequencies.append(0.2)
            if sims[i] >= 0.25:
                frequencies.append(0.25)
            if sims[i] >= 0.3:
                frequencies.append(0.3)
            if sims[i] >= 0.35:
                frequencies.append(0.35)
            if sims[i] >= 0.4:
                frequencies.append(0.4)
            if sims[i] >= 0.45:
                frequencies.append(0.45)                
        if sims[i] >= 0.5:
            frequencies.append(0.5)
        if sims[i] >= 0.55:
            frequencies.append(0.55)
        if sims[i] >= 0.6:
            frequencies.append(0.6)
        if sims[i] >= 0.65:
            frequencies.append(0.65)
        if sims[i] >= 0.7:
            frequencies.append(0.7)
        if sims[i] >= 0.75:
            frequencies.append(0.75)
        if sims[i] >= 0.8:
            frequencies.append(0.8)
        if sims[i] >= 0.85:
            frequencies.append(0.85)
        if sims[i] >= 0.9:
            frequencies.append(0.9)
        if sims[i] >= 0.95:
            frequencies.append(0.95)
        if sims[i] >= 1:
            frequencies.append(1)
        
    return frequencies, sims

# %% Figure for 3.1 Frequencies
frequencies, sims = generate_bernoulli(prob = 0.5)
fig = plt.figure()
plt.hist(frequencies, bins = 9, weights=np.zeros_like(frequencies) + 1. / len(sims), label = "Emperical Frequencies")
plt.title("Emperical Frequencies for coinflip experiment, bias = 0.5")
plt.xlabel(r'$\alpha$')
plt.ylabel("Frequency")
plt.legend(loc="upper left")
plt.legend(loc = "upper right")
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/opg_31.png', bbox_inches='tight')

# %% 3.3 Markov
alphas = np.linspace(0.5, 1.0, 11)
markov_bound = 1/(alphas*2)
fig = plt.figure()
plt.hist(frequencies, bins = 9, weights=np.zeros_like(frequencies) + 1. / len(sims), label = "Emperical Frequencies")
plt.title("Emperical Frequencies with Markov's Bound, bias = 0.5")
plt.xlabel(r'$\alpha$')
plt.ylabel("Frequency")
plt.plot(alphas, markov_bound, label = "Markov's Bound")
plt.xlim(0.48, 1)
plt.legend(loc = "upper right")
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/opg_32.png', bbox_inches='tight')

# %% 3.4 Chebyshevs
chebyshevs_bound = np.minimum(1/(80*(alphas - 0.5)**2), 1)
fig = plt.figure()
plt.hist(frequencies, bins = 9, weights=np.zeros_like(frequencies) + 1. / len(sims), label = "Emperical Frequencies")
plt.title("Emperical Frequencies with Markov's and Chebyshevs Bound, bias = 0.5")
plt.xlabel(r'$\alpha$')
plt.ylabel("Frequency")
plt.plot(alphas, markov_bound, label = "Markov's Bound")
plt.plot(alphas, chebyshevs_bound , label = "Chebyshev's Bound")
plt.xlim(0.48, 1)
plt.legend(loc = "upper right")
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/opg_33.png', bbox_inches='tight')

# %% 3.5 Hoeffdings
frequencies, sims = generate_bernoulli(prob = 0.5)
hoeffdings_bound = np.minimum(np.exp(-2*n*(alphas - 0.5)**2), 1)
fig = plt.figure()
plt.hist(frequencies, bins = 9, weights=np.zeros_like(frequencies) + 1. / len(sims), label = "Emperical Frequencies")
plt.title("Emperical Frequencies with all Bound, bias = 0.5")
plt.xlabel(r'$\alpha$')
plt.ylabel("Frequency")
plt.plot(alphas, markov_bound, label = "Markov's Bound")
plt.plot(alphas, chebyshevs_bound, label = "Chebyshev's Bound")
plt.plot(alphas, hoeffdings_bound, label = "Hoeffding's Bound")
plt.xlim(0.48, 1)
plt.legend(loc = "upper right")
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/opg_34.png', bbox_inches='tight')

# %% 3.7 exact probabilities'
import math
p = 0.5
alpha1 = math.factorial(n)/(math.factorial(20)*math.factorial((20-20)))*p**20*(1-p)**(20-20)
alpha95 = alpha1 + math.factorial(n)/(math.factorial(20-1)*math.factorial((20-19)))*p**(19)*(1-p)**(20-19) 


# %% 3b
frequencies, sims = generate_bernoulli(prob = 0.1)
alphas = np.linspace(0.1, 1.0, 18)
markov_bound = np.minimum(1/(alphas*10),1)
chebyshevs_bound = np.minimum(9/(2000*(alphas - 0.1)**2), 1)
hoeffdings_bound = np.minimum(np.exp(-2*n*(alphas - 0.1)**2), 1)
# %%
fig = plt.figure()
plt.hist(frequencies, bins = 10, weights=np.zeros_like(frequencies) + 1. / len(sims), label = "Emperical Frequencies")
plt.title("Emperical Frequencies with all Bound's', bias = 0.1")
plt.xlabel(r'$\alpha$')
plt.ylabel("Frequency")
plt.plot(alphas, markov_bound, label = "Markov's Bound")        
plt.plot(alphas, chebyshevs_bound, label = "Chebyshev's Bound")
plt.plot(alphas, hoeffdings_bound, label = "Hoeffding's Bound")
plt.xlim(0.08, 1)
plt.legend(loc = "upper right")
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/opg_3b.png', bbox_inches='tight')

# %%
p = 0.1
alpha1 = math.factorial(n)/(math.factorial(20)*math.factorial((20-20)))*p**20*(1-p)**(20-20)
alpha95 = alpha1 + math.factorial(n)/(math.factorial(20-1)*math.factorial((20-19)))*p**(19)*(1-p)**(20-19) 
print(alpha1)
print(alpha95)

# %% 5
x = np.array([1, 2, 3, 4, 5])
y = np.array([14, 21, 25, 35, 32])
x = np.array([x, x**2])
coefs = np.linalg.inv(x @ x.T) @ x @ y.T

x_new = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.8127])
x_new = np.array([x_new, x_new**2])
y_hat = coefs.T @ x_new
fig = plt.figure()
plt.plot(x_new[0,], y_hat)
plt.scatter(x[0,], y)
plt.title("Trajection of canonball")
plt.xlabel("Distance")
plt.ylabel("Height")
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/opg_5.png', bbox_inches='tight')

# %% K NEAREST NEIGHBOUR
digits = np.loadtxt('MNIST-5-6-Subset/MNIST-5-6-Subset.txt', dtype='float')
labels = np.loadtxt('MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt', dtype='int')
digits_light = np.loadtxt('MNIST-5-6-Subset/MNIST-5-6-Subset-Light-Corruption.txt', dtype='float')
digits_moderate = np.loadtxt('MNIST-5-6-Subset/MNIST-5-6-Subset-Moderate-Corruption.txt', dtype='float')
digits_heavy = np.loadtxt('MNIST-5-6-Subset/MNIST-5-6-Subset-Heavy-Corruption.txt', dtype='float')
# %%
digits = np.resize(digits,(1877, 784))
digits_light = np.resize(digits_light,(1877, 784))
digits_moderate = np.resize(digits_moderate,(1877, 784))
digits_heavy = np.resize(digits_heavy,(1877, 784))

# %%
import scipy.misc
from scipy import ndimage
def plot_img(img, savepath):
    example_pic = img
    example_pic = np.resize(example_pic, (28, 28))
    fig = plt.figure()
    plt.imshow(example_pic.T, cmap = 'gray_r')
    plt.colorbar()
    plt.grid(True)
    fig.savefig(savepath, bbox_inches='tight')



plot_img(img = digits[0,], savepath = 'C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/digit_5.png')
plot_img(img = digits_light[0,], savepath = 'C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/digit_5_light.png')
plot_img(img = digits_moderate[0,], savepath = 'C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/digit_5_moderate.png')
plot_img(img = digits_heavy[0,], savepath = 'C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/digit_5_heavy.png')

# %% KNN
def get_k_nearest(x, y, k, lab = labels):
    k_nearest = np.argsort(np.diag((x - y) @ (x - y).T))[:k]
    k_nearest = np.bincount(labels[k_nearest])
    return np.argmax(k_nearest)

def val_err(estimated, actual):
    return np.sum(np.abs(estimated - actual))/len(estimated)
    
def get_k_nearest2(x, y, k, lab = labels):
    k_nearest = np.argsort(np.diag((x - y) @ (x - y).T))[:k]
    k_nearest = np.bincount(labels[k_nearest])
    return np.argmax(k_nearest)

def plot_knn(errors, title = None, n = 10, err_plot = False, corruption = "none"):
    fig = plt.figure()
    plt.plot(range(1,51), errors[0:50], label = "i = 1")
    plt.plot(range(1,51), errors[50:100], label = "i = 2")
    plt.plot(range(1,51), errors[100:150], label = "i = 3")
    plt.plot(range(1,51), errors[150:200], label = "i = 4")
    plt.plot(range(1,51), errors[200:250], label = "i = 5")
    plt.legend(loc = "upper left")
    plt.title("KNN with n = " + str(n) + " and corruption = " + corruption)
    plt.xlabel("K")
    plt.ylabel("Absolute Loss %")
    if title != None:
        plt.savefig(title, bbox_inches='tight')

d1 = get_k_nearest(digits, digits[0,], 3)
import pandas as pd
def run_knn(n, data, labels = labels,plot = True, corruption = "none"):
    digits_estimated = []
    validation_errors = []
    error_var = []
    train = data[:100,]
    for i in range(1, 6):
        val_set = data[100+i*n:100+(i+1)*n,]
        val_labels = labels[100+i*n:100+(i+1)*n,]        
        for k in range(1, 51):
            digits_estimated = []
            for i in range(n):
                digits_estimated.append(get_k_nearest(train, val_set[i], k))
            validation_errors.append(val_err(digits_estimated, val_labels))
    if plot == True:
        plot_knn(validation_errors, 
                 title = 'C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/knn_'+ corruption + '_n_' + str(n) + '.png',
                 n = n,
                 corruption = corruption)

    error_var = np.var(np.array(validation_errors).reshape(5,50), axis = 0)
    
    return {'error_var': error_var,
            'validation_errors': validation_errors}

val_var = []
for i in [10, 20, 40, 80]:
    knn_test = run_knn(i, data=digits, plot=True)
    val_var.append(knn_test['error_var'])

fig = plt.figure()
plt.plot(range(1,51), val_var[0], label = "n = 10")
plt.plot(range(1,51), val_var[1], label = "n = 20")
plt.plot(range(1,51), val_var[2], label = "n = 40")
plt.plot(range(1,51), val_var[3], label = "n = 80")
plt.legend(loc = "upper left")
plt.title("KNN Validation Error Variance")
plt.xlabel("K")
plt.ylabel("Validation Error Variance")
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/'+ f'{digits=}'.split('=')[0] + 'knn_n_var.png', bbox_inches='tight')

    
val_var = []
for i in [80]:
    knn_test = run_knn(i, data=digits_light, plot=True, corruption = 'light')
    val_var.append(knn_test['error_var'])
    
val_var = []
for i in [80]:
    knn_test = run_knn(i, data=digits_moderate, plot=True, corruption = 'moderate')
    val_var.append(knn_test['error_var'])

val_var = []
for i in [80]:
    knn_test = run_knn(i, data=digits_heavy, plot=True, corruption = 'heavy')
    val_var.append(knn_test['error_var'])


