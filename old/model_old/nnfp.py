#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnfp.py: Neural Network Fingerprinter, based on " "


Created on Thu Jun 13 20:35:34 2019

@author: skchang@cochlear.ai
"""
import numpy as np
import tensorflow as tf
assert tf.__version__ >= "2.0"



class ConvLayer(tf.keras.layers.Layer):
    """
    Arguments
    ---------
    hidden_ch: (int)
    
    Input
    -----
    x: (B,F,T,1)
    
    [Conv1x3]>>[ELU]>>[BN]>>[Conv3x1]>>[ELU]>>[BN]
    
    Output
    ------
    x: (B,F,T,C) with {F=F/stride, T=T/stride, C=hidden_ch}
    """
    def __init__(self, hidden_ch=128, strides=[(1,1),(1,1)]):
        super(ConvLayer, self).__init__()
        self.conv2d_1x3 = tf.keras.layers.Conv2D(hidden_ch,
                                                 kernel_size=(1, 3),
                                                 strides=strides[0],
                                                 padding='SAME',
                                                 dilation_rate=(1, 1),
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='zeros')
        self.conv2d_3x1 = tf.keras.layers.Conv2D(hidden_ch,
                                                 kernel_size=(3, 1),
                                                 strides=strides[1],
                                                 padding='SAME',
                                                 dilation_rate=(1, 1),
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='zeros')
        
        self.BN_1x3 = tf.keras.layers.BatchNormalization()
        self.BN_3x1 = tf.keras.layers.BatchNormalization()
    
    
    def call(self, x):
        x = self.conv2d_1x3(x)
        x = tf.nn.elu(x)
        x = self.BN_1x3(x)
        x = self.conv2d_3x1(x)
        x = tf.nn.elu(x)
        return self.BN_3x1(x)
    
    


class DivEncLayer(tf.keras.layers.Layer):
    """
    Arguments
    ---------
    q: (int) number of slices, equivalent with output embedding dim. slice_length = input_sz / q

    Input
    -----
    x: (B,1,1,C)
    
    Output
    ------
    emb: (B,Q)
    """
    def __init__(self, q=128, unit_dim=[32, 1]):
        super(DivEncLayer, self).__init__()

        self.q = q
        self.unit_dim = unit_dim
        self.split_fc_layers = self._construct_layers() 


    def build(self, input_shape):
        # Prepare output embedding variable for dynamic batch-size 
        self.slice_length = int(input_shape[-1] / self.q)
#        self.emb = self.add_variable("emb", shape=[int(input_shape[0]), self.q]) # emb: (B,Q)


    def _construct_layers(self):
        layers = list()
        for i in range(self.q): # q: num_slices
            layers.append(tf.keras.Sequential([tf.keras.layers.Dense(self.unit_dim[0], activation='elu'),
                                               tf.keras.layers.BatchNormalization(),
                                               tf.keras.layers.Dense(self.unit_dim[1])]))
        return layers


#    @tf.function
#    def _split_encoding(self, x_slices):
#        """
#        Input: (B,Q,S)
#        emb:(B,Q) 
#        """
#        for i in range(self.q): # q: num_slices
#            self.emb[:,i].assign(tf.squeeze(self.split_fc_layers[i](x_slices[:, i, :])));
#        return self.emb
    
    @tf.function
    def _split_encoding(self, x_slices):
        """
        Input: (B,Q,S)
        emb:(B,Q) 
        """
        out = list()
        for i in range(self.q):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        return tf.concat(out, axis=1)

    
    
    def call(self, x): # x: (B,1,1,1024)
        x = tf.reshape(x, shape=[x.shape[0], 128, -1]) # (B,Q,S); Q=num_slices; S=slice length; (B,128,8 or 16)
        return self._split_encoding(x)
    
    
    

class FingerPrinter(tf.keras.Model):
    """
    Arguments
    ---------
    front_hidden_ch: (list)
    front_strides: (list)
    emb_sz: (int) default=128
    fc_unit_dim: (list) default=[32,1]
    
    Input
    -----
    x: (B,F,T,1)
        
    Returns
    -------
    emb: (B,Q) 
    """
    def __init__(self,
                 #front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                 front_hidden_ch=[64, 64, 128, 128, 256, 256, 512, 512],
                 front_strides=[[(1,1), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)]],
                 emb_sz=128, # q
                 fc_unit_dim=[32,1]):
        super(FingerPrinter, self).__init__()
        
        self.n_clayers = len(front_strides)
        
        # Front (sep-)conv layers 
        self.front_conv = tf.keras.Sequential()
        for i in range(self.n_clayers):
            self.front_conv.add(ConvLayer(hidden_ch=front_hidden_ch[i],
                                          strides=front_strides[i]))
        self.front_conv.add(tf.keras.layers.Flatten()) # (B,F',T',C) >> (B,D)
        
        # Divide & Encoder layer
        self.div_enc = DivEncLayer(emb_sz, fc_unit_dim)

        
    @tf.function
    def call(self, x):
        x = self.front_conv(x) # (B,D) with D = (T/2^4) x last_hidden_ch
        return self.div_enc(x) # (B,Q)





def test():
    from utils.config_gpu_memory_lim import allow_gpu_memory_growth
    allow_gpu_memory_growth()
    _input = tf.constant(np.random.randn(3,256,32,1), dtype=tf.float32) # BxFxTx1
    fprinter = FingerPrinter()
    fprint = fprinter(_input)
    #%timeit -n 10 fprinter(_input) # 14.6ms
    
    return fprint

    
        
    
        