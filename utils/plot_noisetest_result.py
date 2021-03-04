#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:17:40 2019

@author: sunkyun
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

acc = np.zeros((5, 8)) # train[0,5,10,20,noAug]
# n= noIR
# top1acc, scope=1, test=[0, 0n, 5, 5n, 10, 10n, 20, 20n] 

acc[0, :] = [39.37, 36.37, 56.38, 58.12, 66.88, 70.01, 71.44, 77.86]
acc[1, :] = [39.54, 37.59, 59.33, 61.07, 69.84, 72.01, 73.87, 77.82]
acc[2, :] = [60.07, 58.68, 71.27, 77.17, 74.39, 77.17, 75.09, 78.95]
acc[3, :] = [31.21, 32.55, 53.6, 55.43, 68.75, 70.23, 76.30, 78.47]
acc[4, :] = [32.73, 33.68, 52.91, 54.86, 67.10, 68.58, 76.65, 78.26]


#%%
plt.figure()
#cs = cm.get_cmap('prism', 8)
plt.set_cmap('Blues')
X =  np.arange(5)
plt.bar(X - 0.4, acc[0], width = 0.1)
plt.bar(X - 0.3, acc[1], width = 0.1)
plt.bar(X - 0.2, acc[2], width = 0.1)
plt.bar(X - 0.1, acc[3], width = 0.1)
plt.bar(X + 0.0, acc[4], width = 0.1)
plt.bar(X + 0.1, acc[5], width = 0.1)
plt.bar(X + 0.2, acc[6], width = 0.1)
plt.bar(X + 0.3, acc[7], width = 0.1)

#%%
import pandas as pd

df = pd.DataFrame(data=acc, index=['0dB', '5dB', '10dB', '20dB', 'noAug'],
                  columns=['0dB', '0dB/noIR','5dB', '5dB/noIR',
                           '10dB', '10dB/noIR','20dB', '20dB/noIR'])
ax = df.plot(kind='bar', colormap='Spectral', rot=0)
ax.set_xlabel('SNR(train)')
ax.set_ylabel('Segment-wise Accuracy(%)')
ax.set_title('Effect of SNR for augmentation')
ax.legend(title='SNR(test)')
