# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Online Normalized Temparature Loss for original/replica pairs.
Your batch should be [org1,org2,...org10,rep1,rep2,...rep10].

Created on Fri Oct 23 19:58:05 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf


class OnlineNTxentLoss():
    """Online Normalized Temparature Loss implementation."""

    def __init__(self,
                 n_org=int(),
                 n_rep=int(), 
                 tau=0.05,
                 ):
        """Init."""
        self.n_org = n_org
        self.n_rep = n_rep
        self.tau = tau
        
        """Generate temporal labels and diag masks."""
        self.labels = tf.one_hot(tf.range(n_org), n_org * 2 - 1)
        self.mask_not_diag = tf.constant(tf.cast(1 - tf.eye(n_org), tf.bool))
        
    
    @tf.function 
    def drop_diag(self, x):
        x = tf.boolean_mask(x, self.mask_not_diag)
        return tf.reshape(x, (self.n_org, self.n_org-1))
    
    
    @tf.function 
    def compute_loss(self, emb_org, emb_rep):
        """Ntxent Loss functions.
        
        NOTE1: all input embeddings must be L2-normalized... 
        NOTE2: emb_org and emb_rep must be even number.
        
        Args:
            emb_org: tensor of shape (nO, d), d is dimension of embeddings. 
            emb_rep: tensor of shape (nR, d)        
                    
        Returns:
        """
        ha, hb = emb_org, emb_rep # assert(len(emb_org)==len(emb_rep))
        logits_aa = tf.matmul(ha, ha, transpose_b=True) / self.tau
        logits_aa = self.drop_diag(logits_aa) # modified
        logits_bb = tf.matmul(hb, hb, transpose_b=True) / self.tau
        logits_bb = self.drop_diag(logits_bb) # modified
        logits_ab = tf.matmul(ha, hb, transpose_b=True) / self.tau
        logits_ba = tf.matmul(hb, ha, transpose_b=True) / self.tau
        loss_a = tf.compat.v1.losses.softmax_cross_entropy(
            self.labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.compat.v1.losses.softmax_cross_entropy(
            self.labels, tf.concat([logits_ba, logits_bb], 1))
        return loss_a + loss_b, tf.concat([logits_ab, logits_aa], 1), self.labels

        
        
        
# Unit-test
def test_ntxent_loss():
    feat_dim = 5
    n_org, n_rep = 3, 3   # NOTE: usually n_org = n_rep. because we always prepare pairwise. Batchsize is 6
    tau = 0.05 # temperature
    emb_org = tf.random.uniform((n_org, feat_dim)) # this should be [org1, org2, org3]
    emb_rep = tf.random.uniform((n_rep, feat_dim)) # this should [rep1, rep2, rep3] 
    
    loss_obj = OnlineNTxentLoss(n_org=n_org, n_rep=n_rep, tau=tau)
    loss, simmtx_upper_half, _ = loss_obj.compute_loss(emb_org, emb_rep)
