# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 08:53:00 2021

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

train_data = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/data_random_forests/landsat_train.csv', delimiter =',')
x_train = train_data[:, 1:10]
y_train = train_data[:, 0]
val_data = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/data_random_forests/landsat_validation.csv', delimiter =',')
x_val = val_data[:, 1:10]
y_val = val_data[:, 0]
x_pred = np.loadtxt('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/data_random_forests/landsat_area.csv', delimiter =',')

########## Pt. 1 Training the model
# instantiating the model
model = RandomForestClassifier(max_depth=None, 
                              random_state=0, 
                              bootstrap=True, 
                              max_features=None,
                              n_estimators=10)

# training the model
model.fit(x_train, y_train)

preds = model.predict(x_val)

val_acc = len(preds[preds == y_val]) / len(preds)

print(val_acc)

########## Pt. 2 Applying to landsat_area
area_predicted = model.predict(x_pred)

to_pic = np.resize(area_predicted,(3000,3000))
plt.imshow(to_pic, interpolation='nearest')
#plt.show()
plt.savefig('C:/Users/lucas/OneDrive/Skrivebord/repo/machine_learning_ku/H05/plots/map.png')
