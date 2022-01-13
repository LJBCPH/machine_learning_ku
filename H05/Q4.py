# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:54:58 2021

@author: lucas
"""

import csv

# read flash.dat to a list of lists
datContent = [i.strip().split() for i in open("C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/VStest.dt").readlines()]

# write it as a new CSV file
with open("./flash.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(datContent)

import numpy as np
#sed = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/VStrain.dat', unpack = True)
dt = np.dtype([('time', [('min', np.int64), ('sec', np.int64)]),
               ('temp', float)])
fh = open("C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/VStrain.dt")
data = np.fromfile(fh)

import codecs
file = codecs.open("C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/VStrain.html", "r", "utf-8")
del(file)

