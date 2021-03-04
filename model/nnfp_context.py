#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnfp_context.py: Neural Network Fingerprinter with contextualized embeddings


Created on Wed Oct 28 21:01:16 2020

@author: skchang@cochlear.ai
"""
import tensorflow as tf
import numpy as np
from .transformers.transformer import Transformer
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
                 norm='layer_norm2d', name=None, **kwargs):
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
        x = self.conv2d_1x3(x)
        x = tf.nn.elu(x)
        x = self.BN_1x3(x)
        x = self.conv2d_3x1(x)
        x = tf.nn.elu(x)
        return self.BN_3x1(x)

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

    def build(self, input_shape):
        # Prepare output embedding variable for dynamic batch-size
        self.slice_length = int(input_shape[-1] / self.q)
        self.reshape = tf.keras.layers.Reshape((self.q, self.slice_length), input_shape=input_shape)
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
        return self.concat_out(out)

    def call(self, x): # x: (B,1,1,2048)
        x = self.reshape(x)
        return self._split_encoding(x)

    def compute_output_shape(self, input_shape):
        return([input_shape[0],self.q])




class LocalFP(tf.keras.Model):
    """
    LocalFP:
        - Generating local embeddings z(t) with segment step t.
        - This was FP in unit A
        - L2 normalization is optional in call().

    Arguments
    ---------
    input_shape    : tuple (int), not including the batch size
    emb_sz         : (int) default=128
    front_hidden_ch: (list)
    front_strides  : (list)
    fc_unit_dim    : (list) default=[32,1]
    norm           : 'layer_norm1d' for normalization on Freq axis. (default)
                     'layer_norm2d' fpr normalization on on FxT space
                     'batch_norm' or else, batch-normalization
    projection_layer_type: 'div_enc' or 'fc' or 'fc_no_BN'
    name           : Object name
    trainable      : (bool) False (default)

    Input
    -----
    x: (B,F,T,1) with default (B, 256, 32, 1)

    Returns
    -------
    emb: (B,Q)
    """
    def __init__(self,
                 input_shape=(256,32,1),
                 emb_sz=128, # q
                 front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                 front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)]],
                 fc_unit_dim=[32, 1],
                 norm='layer_norm2d',
                 projection_layer_type='div_enc', # 'div_enc' or 'fc' or 'fc_no_BN'
                 name=None,
                 trainable=False,
                 **kwargs
                ):
        super(LocalFP, self).__init__(name=name, trainable=trainable)
        self.emb_sz = emb_sz
        self.fc_unit_dim = fc_unit_dim
        self.norm = norm
        self.n_clayers = len(front_strides)
        self.projection_layer_type = projection_layer_type

        self.front_conv = tf.keras.Sequential(name='{}/ConvLayers'.format(self.name))
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


    def build(self, input_shape):
        if self.projection_layer_type=='div_enc':
            self.div_enc.build((self.front_conv.output.shape[-1],))


    @tf.function
    def call(self, x=None, apply_L2=False):
        x = self.front_conv(x) # (B,D) with D = (T/2^4) x last_hidden_ch
        x = self.div_enc(x) # (B,Q)

        if apply_L2:
            x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
        return x


    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.emb_sz]



class FingerPrinter(tf.keras.Model):
    """
    FingerPrinter:
        - consists of LocalFP, z=f(g(.)) and ContextFP, c=h(z)
        - LocalFP:
            - Generating local embeddings z(t) with segment step t.
            - This was FP in unit A
            - L2 normalization is optional in call().
        - ContextFP:
            - based on Transformer
            - L2 normalization is optional in call().

    Arguments
    ---------
    emb_sz         : (int) default=128
    trainable      : (bool) default is False.

        Transformer parameters
        ----------------------
        h_use_input_mask : (bool)
        h_context_mode   : 'bidirectional' or 'forward'
        h_num_heads
        h_dff
        h_max_pos_enc

    NOTE: It is important to set 'trainable' argument explicitly.
          Because transformer will apply dropout only if trainable=True.

    Input
    -----
    x: (B_seq, B_seg, F, T, 1) with default (_, _, 256, 32, 1)

    Returns
    -------
    z: (B_seq, B_seg, D)
    c: (B_seq, B_seg, D)


    Description
    -----------
    IN: x(B_seq, B_seg, F, T, 1)  -->  [reshape]  -->  x(B_seq*B_seg, F , T, 1)
    -->  [LocalFP]  -->  (B_seq*B_seg, D)  -->  [reshape]  -->  (B_seq, B_seg, D)
    -->  [ContextFP]  -->  (B_seq,  B_seg or 1, D): OUT

    """
    def __init__(self,
                 emb_sz=128, # q
                 h_num_layers=8,
                 h_use_input_mask=True,
                 h_context_mode='bidirectional',
                 h_num_heads=4,
                 h_dff=512,  # d of FFN
                 h_seg_scope=5,
                 h_max_pos_enc=40,
                 name=None,
                 trainable=False,
                 **kwargs
                ):
        super(FingerPrinter, self).__init__(name=name, trainable=trainable)
        self.emb_sz = emb_sz
        self.h_num_layers = h_num_layers
        self.h_use_input_mask = h_use_input_mask
        self.h_context_mode = h_context_mode
        self.h_seg_scope = h_seg_scope

        self.local_fp = LocalFP(emb_sz=128, fc_unit_dim=[32, 1], trainable=trainable)
        self.context_fp = Transformer(num_layers=h_num_layers,
                                      d_model=emb_sz,
                                      num_heads=h_num_heads,
                                      dff=h_dff,
                                      maximum_position_encoding=h_max_pos_enc,
                                      trainable=trainable)
        self.att_mask = self.generate_attention_mask()


    def build(self, batch_input_shape):
        self.b_seq, self.b_seg, _, _, _ = batch_input_shape
        if self.h_use_input_mask:
            self.in_mask = self.generate_input_mask()


    def generate_input_mask(self):
        """Generate input mask with size (b_seg, emb_sz)."""
        if (self.h_context_mode=='bidirectional'):
            masking_idx = self.b_seg // 2
        elif (self.h_context_mode=='forward'):
            masking_idx = self.b_seg - 1
        else:
            raise NotImplementedError(self.h_context_mode)

        #input_mask = 1 - tf.reshape(tf.one_hot(tf.repeat(2, repeats=b_seq*b_seg), depth=5), (b_seq,b_seg,self.emb_sz))
        im = np.ones((self.b_seg, self.emb_sz), np.float32)
        im[masking_idx, :] = 0.
        return tf.constant(im)


    def generate_attention_mask(self):
        if (self.h_context_mode=='bidirectional'):
            return None
        else:
            raise NotImplementedError(self.h_context_mode)
            return None


    def unpack_sequence_to_segment(self, x):
        shape = tf.shape(x)
        return tf.reshape(x, (shape[0]*shape[1], shape[2], shape[3], 1)) 


    def pack_segment_to_sequence(self, x, b_seq, b_seg):
        return tf.reshape(x, (b_seq, b_seg, self.emb_sz))


    @tf.function
    def call(self, x=None, compute_z_only=False, apply_L2=False): # input: (B0,B1,F,T,1)
        b_seq, b_seg = tf.shape(x)[0], tf.shape(x)[1]
        x = self.unpack_sequence_to_segment(x) # out:(B,F,T,1)
        z = self.local_fp(x, apply_L2=True) # out: (B,D)
        z = self.pack_segment_to_sequence(z, b_seq, b_seg) # out: (B0,B1,D)
        if apply_L2:
            z = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(z)


        if compute_z_only==False:
            # NOTE:Need to fix input_mask for variable length input, later...!!!
            if self.h_use_input_mask:
                z_masked = z * self.in_mask
                c = self.context_fp(z_masked, training=self.trainable, mask=self.att_mask) # out: (B0,B1,D)
            else:
                c = self.context_fp(z, training=self.trainable, mask=self.att_mask) # out: (B0,B1,D)
            if apply_L2:
                c = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(c)
        else:
            c = None

        return z, c


    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.emb_sz]




def test_LocalFP():
    #from utils.config_gpu_memory_lim import allow_gpu_memory_growth
    #allow_gpu_memory_growth()
    _x = tf.random.uniform((3,256,32,1)) # BxFxTx1
    lfp = LocalFP(emb_sz=128, fc_unit_dim=[32, 1], trainable=False)
    _y = lfp(_x, apply_L2=True) # (3,128)
    #%timeit -n 5 lfp(_x) # 13.7ms
    # 16.9M
    return 0


def test_ContextFP():
    _x = tf.random.uniform((10, 40, 128)) # BxTxD
    cfp = Transformer(num_layers=8, d_model=128, num_heads=8,
                       dff=512, maximum_position_encoding=40, trainable=False)
    _y = cfp(_x, training=False, mask=None) # (3,128)
    #%timeit -n 5 cfp(_x, training=False, mask=None) # 20ms
    # 0.26M
    return 0



def test_FingerPrinter():
    _x = tf.random.uniform((7,5,256,32,1)) # input: (B_seq, B_seg, F, T, 1)
    fp = FingerPrinter(emb_sz=128,
                       h_use_input_mask=True, h_context_mode='bidirectional',
                       h_num_heads=4, h_dff=512, h_max_pos_enc=5,
                       trainable=False)
    _z, _c = fp(_x) # output: (B_seq, B_seg, D)
    return 0
