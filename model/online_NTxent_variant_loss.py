# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Fri Apr 10 23:08:46 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf

LARGE_NUM = 1e9#1e9#1e18 #1e9

class OnlineNTxentVariantLoss():
    """Online Normalized Temparature Loss implementation and its variants."""

    def __init__(self,
                 n_org=int(),
                 n_rep=int(), 
                 tau=1.0,
                 rep_org_weights=1.,
                 mode=str(),
                 label_smooth=0):
        """Init."""
        self.n_org = n_org
        self.n_rep = n_rep
        self.n_rep_split = int(n_rep / n_org)
        self.tau = tau
        self.rep_org_weights = rep_org_weights
        self.mode = mode
        self.label_smooth = label_smooth
        
        """Generate temporal labels and diag masks."""
        if mode in ['simCLR_use_rep_only', 'simCLR_use_anc_rep']:
            self.labels = tf.one_hot(tf.range(n_org), n_org * 2)
            self.mask_diag_org = tf.eye(n_org)
        elif mode == 'simCLR_use_anc_rep_mod':
            self.labels = tf.one_hot(tf.range(n_org), n_org * 2 - 1)
            self.mask_not_diag = tf.constant(tf.cast(1 - tf.eye(n_org), tf.bool))
        
        
    @tf.function 
    def batch_NTxent_loss(self, emb_org, emb_rep):
        """Ntxent Loss functions.
        
        NOTE: all input embeddings must be L2-normalized...
        
        Args:
            emb_org: tensor of shape (nO, d), d is dimension of embeddings. 
            emb_rep: tensor of shape (nR, d)        
            mode:
                - 'simCLR_use_rep_only':
                    Implemented as in simCLR official code. 
                    Only works with 2 replicas per each org samples.
                    Emb_org will be ignored.
                    Diagonal elements in sim-mtx will be very small fixed value.
                - 'simCLR_use_anc_rep_mod':
                    Diagonal elements in sim-mtx will be completely dropped in graph.
                    
        Returns:
            if 'simCLR_use_rep_only':

        """
        if self.mode in ['simCLR_use_rep_only', 'simCLR_use_anc_rep']:
            if self.mode == 'simCLR_use_rep_only':
                ha, hb = tf.split(emb_rep, 2, axis=0)
            else:
                ha, hb = emb_org, emb_rep # assert(len(emb_org)==len(emb_rep))
            logits_aa = tf.matmul(ha, ha, transpose_b=True) / self.tau
            logits_aa = logits_aa - self.mask_diag_org * LARGE_NUM
            logits_bb = tf.matmul(hb, hb, transpose_b=True) / self.tau
            logits_bb = logits_bb - self.mask_diag_org * LARGE_NUM
            logits_ab = tf.matmul(ha, hb, transpose_b=True) / self.tau
            logits_ba = tf.matmul(hb, ha, transpose_b=True) / self.tau
            loss_a = tf.compat.v1.losses.softmax_cross_entropy(
                self.labels, tf.concat([logits_ab, logits_aa], 1), 
                weights=self.rep_org_weights)
            loss_b = tf.compat.v1.losses.softmax_cross_entropy(
                self.labels, tf.concat([logits_ba, logits_bb], 1),
                weights=self.rep_org_weights)
            return loss_a + loss_b, tf.concat([logits_ab, logits_aa], 1), self.labels
        elif self.mode in ['simCLR_use_anc_rep_mod']:
            ha, hb = emb_org, emb_rep # assert(len(emb_org)==len(emb_rep))
            logits_aa = tf.matmul(ha, ha, transpose_b=True) / self.tau
            logits_aa = self.drop_diag(logits_aa) # modified
            logits_bb = tf.matmul(hb, hb, transpose_b=True) / self.tau
            logits_bb = self.drop_diag(logits_bb) # modified
            logits_ab = tf.matmul(ha, hb, transpose_b=True) / self.tau
            logits_ba = tf.matmul(hb, ha, transpose_b=True) / self.tau
            loss_a = tf.compat.v1.losses.softmax_cross_entropy(
                self.labels, tf.concat([logits_ab, logits_aa], 1), 
                weights=self.rep_org_weights,
                label_smoothing=self.label_smooth)
            loss_b = tf.compat.v1.losses.softmax_cross_entropy(
                self.labels, tf.concat([logits_ba, logits_bb], 1),
                weights=self.rep_org_weights,
                label_smoothing=self.label_smooth)
            return loss_a + loss_b, tf.concat([logits_ab, logits_aa], 1), self.labels
            
        
        else:
            raise tf.errors.InvalidArgumentError(None, None, f'Undefined mode: {self.mode}')
            return 0
        # elif mode == 'use_rep_only_reduce_diag_v2':
        #     pairwise_sim = tf.matmul(emb_rep, tf.transpose(emb_rep)) # (nR, nR)
        #     pairwise_sim = pairwise_sim * self.mask_reduce_diag_rep # reduce diagonal elements
            
        #     softmax_vertical = tf.nn.softmax(pairwise_sim, axis=0)
        #     softmax_horizonal = tf.nn.softmax(pairwise_sim, axis=1)
        #     softmax_avg = 0.5 * (softmax_vertical + softmax_horizonal)
        #     return loss

        
        return 0
    
    @tf.function
    def drop_diag(self, x):
        x = tf.boolean_mask(x, self.mask_not_diag)
        return tf.reshape(x, (self.n_org, self.n_org-1))
        
        
        
# Unit-test
def test_ntxent_loss():
    feat_dim = 5
    n_org, n_rep = 3, 6
    emb_org = tf.random.uniform((n_org, feat_dim))
    emb_rep = tf.random.uniform((n_rep, feat_dim))
