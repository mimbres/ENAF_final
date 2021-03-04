# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Thu Oct  8 21:58:32 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt 
from .transformer_helper import abs_positional_encoding, EncoderLayer


# NOTE: We omit the embedding layer for the encoder here.    
#       Instead, this encoder takes as input BxTxD tensor.
class Transformer(tf.keras.Model):
    """
    Arguments
    ---------
    num_layers:                (int) number of multi-head-attention layers
    d_model:                   (int) input dimension
    num_heads:                 (int) number of head
    dff:                       (int) linear layer dimension
    maximum_position_encoding: (int) maxmim input length, T
    rate:                      (float) dropout rate 0~1. 
    
    Input
    -----
    x:          tensor with (B,T,D) or (batch_size, input_seq_len, d_model)
    training:   (bool) 
    mask:       
    
    Output
    ------
    x:          tensor with (B,T,D) or (batch_size, input_seq_len, d_model)

    """
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 maximum_position_encoding,
                 rate=0.1,
                 name=None,
                 trainable=False,
                 **kwargs):
        super(Transformer, self).__init__(name=name, trainable=trainable)
        self.d_model = d_model
        self.num_layers = num_layers 
        self.pos_encoding = abs_positional_encoding(maximum_position_encoding, 
                                                self.d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        
          
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1] # x: (batch_size, input_seq_len, d_model)
        #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :] 
        
        for i in range(self.num_layers):
          x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


def unit_test():
    model = Transformer(num_layers=2, d_model=128, num_heads=8, 
                       dff=256, maximum_position_encoding=10000)
    dummy_input = tf.random.uniform((8, 1000, 128)) # BxTxD
    sample_output = model(dummy_input, training=False, mask=None)
    print (sample_output.shape)  # (batch_size, input_seq_len, d_model)   
    return