#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnfp.py: Neural Network Fingerprinter, based on " "


Created on Thu Jun 13 20:35:34 2019
Updated on Fri Apr 17 21:01:16 2020
- replacing LayerNorm with BatchNorm
@author: skchang@cochlear.ai
"""
import numpy as np
import tensorflow as tf
from .coordconv.CoordConv import AddCoords
assert tf.__version__ >= "2.0"

class ConvLayer(tf.keras.layers.Layer):
    """
    Arguments
    ---------
    hidden_ch: (int)
    strides: [( , )( , )]
    norm: 'layer_norm1d' for normalization on Freq axis. (default)
          'layer_norm2d' fpr normalization on on FxT space 
          'batch_norm' or else, batch-normalization
    
    Input
    -----
    x: (B,F,T,1)
    
    [Conv1x3]>>[ELU]>>[BN]>>[Conv3x1]>>[ELU]>>[BN]
    
    Output
    ------
    x: (B,F,T,C) with {F=F/stride, T=T/stride, C=hidden_ch}
    """
    def __init__(self,
                 hidden_ch=128,
                 strides=[(1,1),(1,1)],
                 norm='batch_norm', name=None, **kwargs):
        self.hidden_ch = hidden_ch
        self.norm = norm
        super(ConvLayer, self).__init__(name=name)
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
        
        if norm == 'layer_norm1d':
            self.BN_1x3 = tf.keras.layers.LayerNormalization(axis=-1)
            self.BN_3x1 = tf.keras.layers.LayerNormalization(axis=-1)
        elif norm == 'layer_norm2d':
            self.BN_1x3 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
            self.BN_3x1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
        else:
            self.BN_1x3 = tf.keras.layers.BatchNormalization(axis=-1) # Fix axis: 2020 Apr20
            self.BN_3x1 = tf.keras.layers.BatchNormalization(axis=-1)
            
    def call(self, x):
#         return self.forward(x)
        x = self.conv2d_1x3(x)
        x = tf.nn.elu(x)
        x = self.BN_1x3(x)
        x = self.conv2d_3x1(x)
        x = tf.nn.elu(x)
        return self.BN_3x1(x)
#         return x
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_ch':self.hidden_ch,
            'norm':self.norm,
            })
        return config        


class DivEncLayer(tf.keras.models.Model):
    """
    Arguments
    ---------
    q: (int) number of slices, equivalent with output embedding dim. slice_length = input_sz / q
    unit_dim: [ , ]
    norm: 'layer_norm1d' or 'layer_norm2d' uses 1D-layer normalization on the feature.
          'batch_norm' or else uses batch normalization.

    Input
    -----
    x: (B,1,1,C)
    
    Output
    ------
    emb: (B,Q)
    """
    def __init__(self, q=128, unit_dim=[32, 1], norm='batch_norm', name=None, **kwargs):
        super(DivEncLayer, self).__init__(name=name)

        self.q = q
        self.unit_dim = unit_dim
        self.norm = norm

        
#         if norm in ['layer_norm1d', 'layer_norm2d']:
#             self.BN = [tf.keras.layers.LayerNormalization(axis=-1) for i in range(q)]
#         else:
#             self.BN = [tf.keras.layers.BatchNormalization(axis=-1) for i in range(q)]
            
#         self.split_fc_layers = self._construct_layers() 

    def build(self, input_shape):
        # Prepare output embedding variable for dynamic batch-size 
        self.slice_length = int(input_shape[-1] / self.q)
        self.reshape = tf.keras.layers.Reshape((self.q, self.slice_length), input_shape=input_shape)
#         self.lambda_split = tf.keras.layers.Lambda(lambda t: tf.split(t, self.q, axis=-1))
        self.split_fc_layers = self._construct_layers(self.slice_length) 
        self.concat_out = tf.keras.layers.Concatenate(axis=1)
    
    def _construct_layers(self, slice_length):
        layers = list()
        for i in range(self.q): # q: num_slices
            layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(slice_length,)),
                    tf.keras.layers.Dense(self.unit_dim[0], activation='elu'),
                    tf.keras.layers.Dense(self.unit_dim[1]), 
                ])
            )
        return layers
    
    def _split_encoding(self, x_slices):
        """
        Input: (B,Q,S)
        emb:(B,Q) 
        """
        out = list()
        for i in range(self.q):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
#             out.append(self.split_fc_layers[i](x_slices[i]))
#         return tf.keras.layers.Lambda(lambda t: tf.concat(t, axis=1), output_shape=(self.q,))(out)
        return self.concat_out(out)
    
    def call(self, x): # x: (B,1,1,2048)
#         x = tf.reshape(x, shape=[x.shape[0], self.q, -1]) # (B,Q,S); Q=num_slices; S=slice length; (B,128,8 or 16)
        x = self.reshape(x)
#         x = self.lambda_split(x)
        return self._split_encoding(x)
    
    def compute_output_shape(self, input_shape):
        return([input_shape[0],self.q])
    
    
    

class FingerPrinter(tf.keras.Model):
    """
    Arguments
    ---------
    input_shape: tuple (int), not including the batch size
    front_hidden_ch: (list)
    front_strides: (list)
    emb_sz: (int) default=128
    fc_unit_dim: (list) default=[32,1]
    norm: 'layer_norm1d' for normalization on Freq axis. (default)
          'layer_norm2d' fpr normalization on on FxT space 
          'batch_norm' or else, batch-normalization
    use_L2layer: True (default)
    use_effnet_ver: False(default) will use now-playing conv
                    (int) will use efficient net with the version
                    
    
    Input
    -----
    x: (B,F,T,1)
        
    Returns
    -------
    emb: (B,Q) 
    """
    def __init__(self,
                 input_shape=(256,32,1),
                 front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                 front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)]],
                 emb_sz=128, # q
                 fc_unit_dim=[32,1],
                 norm='batch_norm',
                 use_L2layer=True,
                 projection_layer_type='div_enc', # 'div_enc' or 'fc' or 'fc_no_BN'
                 add_prediction_layer=False, # This is only for BYOL
                 add_coordconv=False, # CoordConv (2020/09/09)
                 name=None,
                 **kwargs
                ):
        super(FingerPrinter, self).__init__(name=name)
        self.emb_sz = emb_sz
        self.fc_unit_dim = fc_unit_dim
        self.norm = norm
        self.use_L2layer = use_L2layer
        self.n_clayers = len(front_strides)
        self.projection_layer_type = projection_layer_type
        self.add_prediction_layer = add_prediction_layer
        self.add_coordconv = add_coordconv

        self.front_conv = tf.keras.Sequential(name='{}/ConvLayers'.format(self.name))
        if self.add_coordconv :
            self.front_conv.add(tf.keras.layers.Input((input_shape[0], input_shape[1], 4))) # 3 if with_r=False
        else:
            self.front_conv.add(tf.keras.layers.Input(input_shape))
        # Fixed 2019.10.04 for variable embedding dimension
        if ((front_hidden_ch[-1] % emb_sz) != 0):
            front_hidden_ch[-1] = ((front_hidden_ch[-1]//emb_sz) + 1) * emb_sz                
        # Front (sep-)conv layers
        for i in range(self.n_clayers):
            self.front_conv.add(ConvLayer(hidden_ch=front_hidden_ch[i],
                strides=front_strides[i], norm=norm))
        self.front_conv.add(tf.keras.layers.Flatten()) # (B,F',T',C) >> (B,D)

        # Divide & Encoder layer
        if self.projection_layer_type=='div_enc': 
            self.div_enc = DivEncLayer(q=self.emb_sz,
                                       unit_dim=self.fc_unit_dim,
                                       norm=self.norm,
                                       name='{}/projection_layer'.format(self.name))
        elif self.projection_layer_type=='fc':
            self.div_enc = tf.keras.Sequential([tf.keras.layers.Dense(512, activation='elu'),
                                               tf.keras.layers.LayerNormalization(axis=-1),
                                               tf.keras.layers.Dense(emb_sz)],
                                           name='{}/projection_layer'.format(self.name))
        elif self.projection_layer_type=='fc_no_BN':
            self.div_enc = tf.keras.Sequential([tf.keras.layers.Dense(512, activation='elu'),
                                               tf.keras.layers.Dense(emb_sz)],
                                           name='{}/projection_layer'.format(self.name))
        
        if self.add_prediction_layer:
            self.prediction_layer = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(self.emb_sz,)), 
                tf.keras.layers.Dense(units=1024, activation=None), 
                tf.keras.layers.BatchNormalization(axis=1),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dense(units=emb_sz, activation=None)
#                 ,tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))
            ], name='{}/prediction_layer'.format(self.name))
        
        if self.add_coordconv:
            self.add_coord = AddCoords(x_dim=input_shape[0],
                                       y_dim=input_shape[1],
                                       with_r=True,
                                       skiptile=True)  
            
        def build(self, input_shape):
            self.div_enc.build((self.front_conv.output.shape[-1],))
        
#     @tf.function
    def call(self, inputs):
        if self.add_coordconv:
            x = self.add_coord(inputs)
        else:
            x = inputs
            
        x = self.front_conv(x) # (B,D) with D = (T/2^4) x last_hidden_ch
        x = self.div_enc(x) # (B,Q)

        if self.add_prediction_layer:
            x = self.prediction_layer(x)
            
        if self.use_L2layer:
            # x = tf.math.l2_normalize(x, axis=1) # FIXED 2019.09.27, before axis was not defined
            x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x) # FIXED 2019.09.27, before axis was not defined            
        return x

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.emb_sz]


def test():
    from utils.config_gpu_memory_lim import allow_gpu_memory_growth
    allow_gpu_memory_growth()
    _input = tf.constant(np.random.randn(3,256,32,1), dtype=tf.float32) # BxFxTx1
    fprinter = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1], norm='layer_norm2d', add_coordconv=True)
    # fprinter = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1],
    #                          norm='layer_norm2d', use_effnet_ver=4)
    fprint = fprinter(_input) # (3,128)
    #%timeit -n 10 fprinter(_input) # 27.9ms
    # 14M
    return fprint
"""
<default with div_enc>
Total params: 16,939,008
Trainable params: 16,939,008
Non-trainable params: 0

<add_coordconv=True>
Total params: 16,940,160
Trainable params: 16,940,160

<256,256,>
Total params: 19,448,960
Trainable params: 19,448,960


use_effnet=False
Total params: 19,224,576
Trainable params: 19,224,576
Non-trainable params: 0

use_effnet=True
Total params: 17,738,616
Trainable params: 17,613,416
Non-trainable params: 125,200
"""

        
    
        
