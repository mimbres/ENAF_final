# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Tue Aug  4 16:31:24 2020
@author: skchang@cochlear.ai
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

softmax = tf.keras.layers.Softmax()
bsz_set = [10, 50]
tau_set = [0.05, 0.1, 0.3]

plt.figure()
legend_str = []

for bsz in bsz_set:
    for tau in tau_set:
        x = np.linspace(-.5, .5, num=bsz)
        y = softmax(x / tau).numpy()
        plt.plot(x, y)
        legend_str.append(f'bsz={bsz}, tau={tau}')
plt.legend(legend_str)    
plt.xlabel('x')
plt.ylabel('softmax( x / tau )')
