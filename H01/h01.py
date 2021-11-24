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

def flip_coin(num_coins):
    coin_sum = 0
    for i in range(num_coins):
        flip = np.random.uniform(0, 1, 1)
        if flip >= 0.5:
            coin_sum += 1
    
    return coin_sum
    
## Flipping 1e6 times 20 coins
sims = ([])
for _ in range(sim):
    sims.append(flip_coin(n)/n)
    
# %%
alpha_list = ([])
for i in range(sim):
    if sims[i] >= 0.5:
        alpha_list.append(0.5)
    if sims[i] >= 0.55:
        alpha_list.append(0.55)
    if sims[i] >= 0.6:
        alpha_list.append(0.6)
    if sims[i] >= 0.65:
        alpha_list.append(0.65)
    if sims[i] >= 0.7:
        alpha_list.append(0.7)
    if sims[i] >= 0.75:
        alpha_list.append(0.75)
    if sims[i] >= 0.8:
        alpha_list.append(0.8)
    if sims[i] >= 0.85:
        alpha_list.append(0.85)
    if sims[i] >= 0.9:
        alpha_list.append(0.9)
    if sims[i] >= 0.95:
        alpha_list.append(0.95)
    if sims[i] >= 1:
        alpha_list.append(1)



# %% Figure for 3.1
fig = plt.figure()
plt.hist(alpha_list, bins = 10, weights=np.zeros_like(alpha_list) + 1. / len(alpha_list))
plt.title("Emperical Frequencies for coinflip experiment, bias = 0.5")
plt.xlabel(r'$\alpha$')
plt.ylabel("No. of observations")
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H01/plots/opg_31.png', bbox_inches='tight')

# %%
print(np.array(alpha_list)/sim)





