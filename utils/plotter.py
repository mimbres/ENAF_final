# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Tue Apr 14 17:25:32 2020
@author: skchang@cochlear.ai
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import io

def save_imshow(x, save_fpath, title=None):
    """Save imshow() as .png file.
    
    x: 2D numpy array
    save_fp: (str) 
    """
    # fig, (ax1) = plt.subplots(1,1)
    # ax1.imshow(x, origin='upper')
    fig = plt.figure()
    plt.set_cmap('Blues')
    plt.imshow(x)
    plt.colorbar()
    if title is not None:
        fig.suptitle(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_fpath)
    plt.close('all')
    print(f'SAVE_IMSHOW: saved image to {save_fpath}...')
    return


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


# This method is useful for displaying imshow in tensorboard
def get_imshow_image(x, title=None):
    """Save imshow() as .png file.
    
    x: 2D numpy array 
    """
    # fig, (ax1) = plt.subplots(1,1)
    # ax1.imshow(x, origin='upper')
    fig = plt.figure()
    plt.set_cmap('Blues')
    plt.imshow(x)
    plt.colorbar()
    if title is not None:
        fig.suptitle(title, fontsize=8)
    
    plt.tight_layout()
    image = plot_to_image(fig)
    plt.close('all')
    print('GET_IMSHOW: created an image for tensorboard...')
    return image