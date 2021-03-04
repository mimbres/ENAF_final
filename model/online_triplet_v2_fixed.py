#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define functions to create the triplet loss with online triplet mining.
Based on "https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py"

Created on Thu Jun 20 19:58:36 2019
Last Update: 2019.10.09

@author: skchang@cochlear.ai
"""
import tensorflow as tf
import numpy as np


class Online_Batch_Triplet_Loss():
    def __init__(self,
                 bsz=int(),
                 n_anchor=int(),
                 n_pos_per_anchor=int(),
                 use_anc_as_pos=True):
        super(Online_Batch_Triplet_Loss, self).__init__()
        
        # Variables
        self.bsz = bsz # Batch size
        self.n_anchor = n_anchor # Total number of anchor samples in batch
        self.n_pos_per_anchor = n_pos_per_anchor # Number of positive samples per each anchor
        self.use_anc_as_pos = use_anc_as_pos # Using each original anchor sample as a member of positive samples
        
        # Anchor-positive & Anchor-engative masks
        self.ap_mask = self._get_anchor_positive_mask_v2()
        self.an_mask = self._get_anchor_negative_mask_v2()
        #self.ap_mask_bin = tf.cast(self.ap_mask, tf.bool)
        self.an_mask_bin = tf.cast(self.an_mask, tf.bool)
        self.gt_mask = tf.abs(1 - self.an_mask) 
        self.mask_shape = self.ap_mask.shape
        
        # Number of positive and negative elements per each anchor (to be used as a normalization factor)
        self.num_ap_elem_per_anc = tf.constant(np.sum(self.ap_mask, axis=1).astype(np.float32)) # (A,)
        self.num_an_elem_per_anc = tf.constant(np.sum(self.an_mask, axis=1).astype(np.float32)) # (A,)


    #---- Numpy functions for generating masks --------------------------------
    def _get_anchor_positive_mask_v2(self):
        n_pos = self.n_anchor * self.n_pos_per_anchor # Here, n_pos does not count anchor-anchor datapoints.
        
        if self.use_anc_as_pos:
            mask = np.zeros((self.n_anchor, n_pos + self.n_anchor)) # (nA, bsz)
        else:
            mask = np.zeros((self.n_anchor, n_pos))
        
        for anchor in range(self.n_anchor):
            mask[anchor, anchor * self.n_pos_per_anchor: (anchor + 1) * self.n_pos_per_anchor] = 1    
        return tf.constant(mask.astype(np.float32))
    
    
    def _get_anchor_negative_mask_v2(self):
        mask = self._get_anchor_positive_mask_v2()
        
        if self.use_anc_as_pos:
            mask = tf.concat((mask[:, :self.n_anchor * self.n_pos_per_anchor], tf.eye(self.n_anchor)), axis=1)
        return (1 - mask)
        

    
    # ---- Tensorflow functions for calculation of pariwise distances --------------------------
    @tf.function
    def _pairwise_distances_v2(self, emb_anc, emb_pos, use_anc_as_pos=True, squared=False):
        """Compute the 2D matrix of distances between anchors and positive embeddings.
        Args:
            emb_anc: tensor of shape (nA, Q)
            emb_pos: tensor of shape (nP, Q)
            NOTE: {emb_anc, emb_pos} must be L2-normalized vectors with axis=1. 
            
        Returns:
            if use_anc_as_pos:
                pairwise_distances: tensor of shape (nA, nP + nA)
            else:
                pairwise_distances: tensor of shape (nA, nP)
        """
        if use_anc_as_pos:
            emb_pos = tf.concat((emb_pos, emb_anc), axis=0) # (A+P, Q)
        else:
            pass;
        dot_product = tf.matmul(emb_anc, tf.transpose(emb_pos)) # (A, A+P)
        
        # Get squared L2 norm for each embedding.
        a_sq = tf.reduce_sum(tf.square(emb_anc), axis=1)# (A, 1) 
        p_sq = tf.reduce_sum(tf.square(emb_pos), axis=1)# (P, 1) or (A+P, 1)
        
        """ Pairwise squared distance matrix:        
        
            ||a - b||^2 = ||a||^2  + ||b||^2 - - 2 <a, b>
        
        """
        
        dists = (tf.expand_dims(a_sq, 1) + tf.expand_dims(p_sq, 0)) - 2.0 * dot_product
        dists = tf.maximum(dists, 0.0) # Make sure every dist >= 0
    
        if not squared:
            # To prevent -inf gradients where dist=0.0, we add small epsilons
            mask = tf.cast(tf.equal(dists, 0.0), tf.float32)
            dists = dists + mask * 1e-16
            dists = tf.sqrt(dists)
            dists = dists * (1.0 - mask)
        return dists


    @tf.function
    def _pairwise_dotprod(self, emb_anc, emb_pos, use_anc_as_pos=True):
        """Compute the 2D matrix of cosine-similarity(=dot-product) between
        anchors and positive embeddings. (2019.10.09)
        
        Args:
            emb_anc: tensor of shape (nA, d), d is dimension of embeddings
            emb_pos: tensor of shape (nP, d)
            NOTE: {emb_anc, emb_pos} must be L2-normalized vectors with axis=1.     
        
        Returns:
            if use_anc_as_pos:
                pairwise_distances: tensor of shape (nA, nP + nA)
            else:
                pairwise_distances: tensor of shape (nA, nP)
        """
        if use_anc_as_pos:
            emb_pos = tf.concat((emb_pos, emb_anc), axis=0) # (A+P, Q)
        else:
            pass;
        
        return tf.matmul(emb_anc, tf.transpose(emb_pos))


    @tf.function
    def _pairwise_distances_v2_fast(self, emb_anc, emb_pos, use_anc_as_pos=True, squared=False):
        dists = self._pairwise_dotprod(emb_anc=emb_anc,
                                     emb_pos=emb_pos,
                                     use_anc_as_pos=use_anc_as_pos)
        dists = 2. * (1 - dists)
        if not squared:
            # To prevent -inf gradients where dist=0.0, we add small epsilons
            mask = tf.cast(tf.equal(dists, 0.0), tf.float32)
            dists = dists + mask * 1e-16
            dists = tf.sqrt(dists)
            dists = dists * (1.0 - mask)
        
        return dists
        
        

    # ---- Loss Functions -----------------------------------------------------
    @tf.function
    def batch_triplet_loss_v2(self, emb_anchor, emb_pos, mode, margin=1.0,
                              use_anc_as_pos=True, squared=False):
        """
        mode:
            "all"
            "all-balanced"
            "hardest"
            "semi-hard"
            
        """
        # Get a pairwise distance matrix
        #pairwise_dist = self._pairwise_distances_v2(emb_anchor, emb_pos, use_anc_as_pos, squared) # (A, P)
        pairwise_dist = self._pairwise_distances_v2_fast(emb_anchor, emb_pos, use_anc_as_pos, squared)
        # Calculate Pos/Neg distances
        ap_dists = pairwise_dist * self.ap_mask
        
        if mode == "all":
            an_dists = pairwise_dist * self.an_mask
            loss = tf.maximum(ap_dists - an_dists + margin, 0.)
            loss = tf.reduce_mean(loss)
        elif mode == "all-balanced":
            ap_dists = tf.divide(tf.reduce_sum(ap_dists, axis=1), self.num_ap_elem_per_anc) 
            an_dists = pairwise_dist * self.an_mask
            an_dists = tf.divide(tf.reduce_sum(an_dists, axis=1), self.num_an_elem_per_anc)
            loss = tf.maximum(ap_dists - an_dists + margin, 0.)
            loss = tf.reduce_mean(loss)
        elif mode == "hardest":
            ap_dists = tf.reduce_max(ap_dists, axis=1) 
            an_dists = tf.reduce_min(pairwise_dist * self.an_mask, axis=1)
            loss = tf.maximum(ap_dists - an_dists + margin, 0.)
            loss = tf.reduce_mean(loss)
        elif mode == "semi-hard":
            # ap_dists: Tiled hardest anchor-positive distances.
            ap_dists = tf.reduce_max(ap_dists, axis=1, keepdims=True) * tf.ones([1, self.mask_shape[1]])
            loss = (ap_dists - pairwise_dist + margin) * self.an_mask
            loss = tf.maximum(loss, 0.) # Neglect easy triplets
            #num_active_triplets = tf.reduce_sum(tf.cast(tf.greater(loss, 0.), tf.float32))
            loss = tf.reduce_mean(loss) #/ num_active_triplets # <-- fixed 09.27
        else:
            raise NotImplementedError(mode)

        return loss 
    
    

    @tf.function
    def batch_cos_loss(self, emb_anchor, emb_pos, margin=0., use_anc_as_pos=True):
        # Calculate pairwise dot-product
        pairwise_dotprod = self._pairwise_dotprod(emb_anc=emb_anchor, emb_pos=emb_pos,
                                             use_anc_as_pos=use_anc_as_pos) # (nClass, batch)
        
        # Transpose to (batch, nClass)
        pairwise_dotprod = tf.transpose(pairwise_dotprod, perm=(1,0))
        gt_mask = tf.transpose(self.gt_mask, perm=(1,0)) # binary mask
        
        # Apply margin
        if margin > 0.0001:
            pairwise_dotprod = pairwise_dotprod - (margin * gt_mask) 
        
        loss = tf.nn.softmax_cross_entropy_with_logits(gt_mask, pairwise_dotprod, axis=-1)
        return tf.reduce_mean(loss)


    @tf.function
    def batch_cos_loss_fix(self, emb_anchor, emb_pos, margin=0., use_anc_as_pos=True):
        # Calculate pairwise dot-product
        pairwise_dotprod = self._pairwise_dotprod(emb_anc=emb_anchor, emb_pos=emb_pos,
                                             use_anc_as_pos=use_anc_as_pos) # (nClass, batch)
        
        # Transpose to (batch, nClass)
        pairwise_dotprod = tf.transpose(pairwise_dotprod, perm=(1,0))
        gt_mask = tf.transpose(self.gt_mask, perm=(1,0)) # binary mask
        
        # Apply margin
        if margin > 0.0001:
            pairwise_dotprod = pairwise_dotprod + (margin * tf.transpose(self.an_mask, perm=(1,0))) 
        
        loss = tf.nn.softmax_cross_entropy_with_logits(gt_mask, pairwise_dotprod, axis=-1)
        return tf.reduce_mean(loss)


    @tf.function
    def batch_cos_semihard_loss(self, emb_anchor, emb_pos, margin=0., use_anc_as_pos=True, use_hard_an=False):
        # Calculate pairwise dot-product
        pairwise_dotprod = self._pairwise_dotprod(emb_anc=emb_anchor, emb_pos=emb_pos,
                                             use_anc_as_pos=use_anc_as_pos) # (nClass, batch)
        
        # {Semi-hard-neg, pos} selection mask: binaray mask for selecting semi-hard triplet pairs, 
        #                                      where neg > pos - margin
        # step1) pos selection
        ap_dotprod = pairwise_dotprod * self.gt_mask # Here, we use gtmask instead of apmask (required for later process)
        
        # step2) neg selection
        hardest_ap = tf.reduce_min(ap_dotprod, axis=1, keepdims=True) * tf.ones([1, self.mask_shape[1]])
        _tmp = tf.greater(pairwise_dotprod, hardest_ap - margin)
        if use_hard_an==False: # If False, this yields semihard-loss
            _tmp = tf.logical_and(tf.less(pairwise_dotprod, hardest_ap), _tmp)
            
        _tmp = tf.logical_and(_tmp, self.an_mask_bin) # logical_selection_mask_for_negatives
        sel_an_mask = tf.cast(_tmp, tf.float32)
        sel_ap_an_mask = self.gt_mask + sel_an_mask
        """
          0.8 - 0.5 + 0.2 = 0.5 (hard)
          0.6 - 0.5 + 0.2 = 0.3 (semihard)
          0.4 - 0.5 + 0.2 = 0.1 (semihard)
          0.1 - 0.5 + 0.2 = -0.2 (good) 
        min(an_sim - hardest_ap_sim + margin, 0.)
         min((neg_sim - hardest_pos_sim), 0)
        0 < (neg - (hardpos - margin)) < margin , with 0 < margin 
        0 < neg - hardpos + margin < margin
        1) neg > hardpos - margin
        2) neg < hardpos
        hardpos - margin < neg < hardpos
        
        neg - hardpos + margin > 0
        neg - hardpos < 0
        
        IFF:        
        -margin < neg-hardpos < 0
        0 < hardpos-neg < margin
        
        -hardpos < -neg < margin-hardpos
        hardpos - margin < neg < hardpos
        
        neg_mask1 = tf.greater(neg - hardpos + margin, 0) 
        """
        pairwise_dotprod = pairwise_dotprod * sel_ap_an_mask
        
        # Now, -inf to ignore unselected logits
        #neg_infs = (1 - sel_ap_an_mask) * tf.float32.min
        neg_infs = (1 - sel_ap_an_mask) * (-1.)
        pairwise_dotprod = pairwise_dotprod + neg_infs
        
        
        # Transpose to (batch, nClass) 
        pairwise_dotprod = tf.transpose(pairwise_dotprod, perm=(1,0))
        gt_mask = tf.transpose(self.gt_mask, perm=(1,0)) # binary mask
        
        loss = tf.nn.softmax_cross_entropy_with_logits(gt_mask, pairwise_dotprod, axis=-1)
        return tf.reduce_mean(loss)
    



def test_mask():
    import matplotlib.pyplot as plt
    OBTL = Online_Batch_Triplet_Loss(bsz=40, n_anchor=8, n_pos_per_anchor=4,
                                     use_anc_as_pos=True)
    ap_mask = OBTL._get_anchor_positive_mask_v2()
    plt.figure()
    plt.imshow(ap_mask); plt.title('Anchor-positive-mask: bsz=40, nAnchor=8, nPosPerAnchor=4')

    an_mask = OBTL._get_anchor_negative_mask_v2()
    plt.figure()
    plt.imshow(an_mask); plt.title('Anchor-negative-mask: bsz=40, nAnchor=8, nPosPerAnchor=4')
    


def test_pair_dist():
    emb_anc = tf.random.uniform((8,64))
    emb_pos = tf.random.uniform((32,64))
    emb_anc = tf.math.l2_normalize(emb_anc, axis=1)
    emb_pos = tf.math.l2_normalize(emb_pos, axis=1)
    
    OBTL = Online_Batch_Triplet_Loss(bsz=40, n_anchor=8, n_pos_per_anchor=4, use_anc_as_pos=True)
    dist1 = OBTL._pairwise_distances_v2(emb_anc, emb_pos, use_anc_as_pos=True, squared=True) 
    dist2 = 2 * (1 - OBTL._pairwise_dotprod(emb_anc, emb_pos, use_anc_as_pos=True))
    dist3 = OBTL._pairwise_distances_v2_fast(emb_anc, emb_pos, use_anc_as_pos=True, squared=True)  
    
    assert(tf.reduce_sum(dist1-dist2) < 0.0000001)
    assert(tf.reduce_sum(dist1-dist3) < 0.0000001)
    return       
    # dist1: L2 distance 19.2ms
    # dist2: fast L2 with dot-product 9.26ms