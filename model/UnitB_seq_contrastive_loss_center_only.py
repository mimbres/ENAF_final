# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Thu Oct 29 21:16:16 2020
@author: skchang@cochlear.ai
"""
import tensorflow as tf
from .online_NTxent_variant_loss import OnlineNTxentVariantLoss # SimCLR loss

LARGE_NUM = 1e9

class ContextContrastiveLoss():
    """Normalized Temparature Loss for sequence prediction with org/rep.
    
    Description
    -----------
    - context_mode=='biredctional':
        
        For each view b and scope=3, we have context-embeddings C:
            {c0_t0, c0_t1, c0_t2, c1_t0, c1_t1, c1_t2,...},
        and its corresponding local-emb Z:
            {z0_t0, z0_t1, z0_t2, z1_t0, z1_t1,...}.
        
        We take sim(c0_t1, z0_t1) as sim(c0, z0), and sim(c1_t1, z1_t1) as sim(c1, z1).
        
        We denote these similairty of positive pairs by anchoring center-context C as
            sim(c0, z0), sim(c1, z1),....    
        (OPTIONAL) We can also use center-local Z as anchor (unlike wav2vec-v2)
             sim(z0,c0), sim(z1,c1)
            
        We denote all other zs who never get a chance to be anchor as [zr] with
            [zr] = {z0_t0, z0_t2, z1_t0, z1_t2,...},
        and [zr] can be used as negative elements.
     
        Finally, `context_loss` with anchor C is calculated as:

            z1 z2 z1` z2`| [zr: rest of center-inactive z]  
        c1  *  -  +   -  | - - - - - - - - - - - - - - - 
        c2  -  *  -   +  | - - - - - - - - - - - - - - - 
        c1` +  -  *   -  | - - - - - - - - - - - - - - - 
        c2` -  +  -   *  | - - - - - - - - - - - - - - - 
        
            1) We generate a sim-mtx A with shape(nC, nZ). 
               Here nZ=Batchsize, and nC=nZ/scope.
            2) For each row, we first let * as positive, and mask +.
            3) For each row, we nextly let + as positive, and mask *.
            
        
    - add_mirror_sim==True:
        
        For each sim(ci,zi), we can add similarity sim(z1,c1) by mirrororing
        across diagonal. Then we can average them.
        We implement it by transposing the sim-mtx A, and recalculate the loss.
        
          
    
    Arguments
    ---------
    
    context_mode    :'bidirectional' or 'forward'(not implemented).
    add_mirror_sim  :(bool) False as default.
    scope           :(int) number of max context scope for transformer. 5 as default.
    τ               :(float) temperature parameter > 0.
    seg_scope       :(int) 
    b_seq           :(int)
    b_seg           :(int)
    weight_context_simclr_loss :[(float),(float)] [context, simclr] add simclr_loss with weight
    
        
    Input
    -----
    z: local representation z of shape (B_seq, B_seg, D)
    c: contextualized representation c of shape (B_seq, B_seg, D) 
    
    
    Returns
    -------    
    context_loss:
        
     
    """
    def __init__(self,
                 context_mode='bidirectional',
                 add_mirror_sim=False,
                 τ=0.05,
                 seg_scope=5,
                 b_seq=60,
                 b_seg=5,
                 weight_context_simclr_loss=[1., 1.]):
        """Init."""
        self.context_mode = context_mode
        self.add_mirror_sim = add_mirror_sim
        self.τ = τ
        self.seg_scope = seg_scope 
        self.b_seq = b_seq
        self.b_seg = b_seg
        self.weight_context_simclr_loss = weight_context_simclr_loss
        
        
        """Add simCLR loss"""
        if (self.weight_context_simclr_loss[1] > 0.):
            self.simclr_loss_func = OnlineNTxentVariantLoss(
                n_org=int(b_seq * b_seg / 2),
                n_rep=int(b_seq * b_seg / 2),
                tau=τ,
                mode='simCLR_use_anc_rep')
            
            
        
        """Generate labels and masks."""
        if (context_mode=='bidirectional'):
            # Count (centered-)active segments in batch
            self.center_start = self.seg_scope // 2 # 2 for scope=5
            self.center_end = self.b_seg - self.center_start
            num_active_seg_in_seq = self.center_end - self.center_start
            self.num_active_seg_all = num_active_seg_in_seq * self.b_seq
            
            # Generate labels
            self.labels_asterisk = tf.one_hot(tf.range(self.num_active_seg_all), self.b_seq * self.b_seg) # (B_center, B_all)
            self.labels_plus = tf.roll(self.labels_asterisk, self.num_active_seg_all//2, axis=0)
            
            # Generate masks
            self.mask_asterisk = tf.eye(self.num_active_seg_all) # (B_center, B_center)
            self.mask_plus = tf.roll(self.mask_asterisk, self.num_active_seg_all//2, axis=0)            
        
        
    @tf.function 
    def compute_loss(self, emb_z, emb_c):
        """Context_contrastive_loss similar with wav2vec-v2.
        
        NOTE: all input embeddings must be L2-normalized...
        
        Args:
            emb_z: tensor of shape (Bseq, Bseg, D).
                   (:Bseq/2,:,:) is original, and (Bseq/2:,:,:) is replica. 
            emb_c: tensor of shape (Bseq, Bseg, D)        
                   (:Bseq/2,:,:) is original, and (Bseq/2:,:,:) is replica. 
                   
        Returns:
            loss
        """
        # get dimension of embeddings, D
        d = tf.shape(emb_z)[-1]
        
        # Select c from the first center to last center in timesteps.
        center_start = self.center_start
        center_end = self.center_end
        
        c = emb_c[:, center_start:center_end, :] 
        c = tf.reshape(c, (-1, d)) # (Bseq*l, D) where l is center-active length
        
        # Unroll and arrange z to have zs from the center in the leftest
        z_center = emb_z[:, center_start:center_end, :]
        z_center = tf.reshape(z_center, (-1, d)) # (Bseq*l, D) where l is center-active length
        z_rest = tf.concat((emb_z[:, :center_start, :],
                            emb_z[:, center_end:, :]), axis=1)
        z_rest = tf.reshape(z_rest, (-1, d)) #  (Bseq*m, D) where m is center-inactive length     
        
        # Compute pair-wise similarity A:
        a_center = tf.matmul(c, z_center, transpose_b=True) / self.τ # (B_center, B_center)
        a_rest   = tf.matmul(c, z_rest, transpose_b=True) / self.τ # (B_center, B_rest)
        """
           <--a_center--> <------------a_rest------------>
            z1 z2 z1` z2`| [zr: rest of center-inactive z]  
        c1  *  -  +   -  | - - - - - - - - - - - - - - - 
        c2  -  *  -   +  | - - - - - - - - - - - - - - - 
        c1` +  -  *   -  | - - - - - - - - - - - - - - - 
        c2` -  +  -   *  | - - - - - - - - - - - - - - - 
        """
        
        # Compute loss from context_prediction as wav2vec-v2 paper.
        logits_asterisk = a_center - self.mask_plus * LARGE_NUM # (B_center, B_center)
        logits_plus     = a_center - self.mask_asterisk * LARGE_NUM # (B_center, B_center)
        
        loss_asterisk = tf.nn.softmax_cross_entropy_with_logits(
            self.labels_asterisk, tf.concat((logits_asterisk, a_rest), axis=1))
        loss_plus = tf.nn.softmax_cross_entropy_with_logits(
            self.labels_plus, tf.concat((logits_plus, a_rest), axis=1))
        loss = loss_asterisk + loss_plus
        
        # (OPTIONAL) Averaging loss by mirrororing across diagonal 
        if self.add_mirror_sim==True:
            # Compute pair-wise similarity A^T
            """
                <--ma_center-> <-----------ma_rest----------->
                c1 c2 c1` c2`| [zr: rest of center-inactive z]  
            z1  *  -  +   -  | - - - - - - - - - - - - - - - 
            z2  -  *  -   +  | - - - - - - - - - - - - - - - 
            z1` +  -  *   -  | - - - - - - - - - - - - - - - 
            z2` -  +  -   *  | - - - - - - - - - - - - - - - 
            """   
            ma_center = tf.matmul(z_center, c, transpose_b=True) / self.τ # (B_center, B_center)
            ma_rest   = tf.matmul(z_center, z_rest, transpose_b=True) / self.τ # (B_center, B_rest)
            
            m_logits_asterisk = ma_center - self.mask_plus * LARGE_NUM # (B_center, B_center)
            m_logits_plus     = ma_center - self.mask_asterisk * LARGE_NUM # (B_center, B_center)
            
            m_loss_asterisk = tf.nn.softmax_cross_entropy_with_logits(
                self.labels_asterisk, tf.concat((m_logits_asterisk, ma_rest), axis=1))
            m_loss_plus = tf.nn.softmax_cross_entropy_with_logits(
                self.labels_plus, tf.concat((m_logits_plus, ma_rest), axis=1)) 
            loss+= m_loss_asterisk + m_loss_plus
        
        
        # (OPTIONAL) Add SimCLR loss: employing every z as an anchor
        if (self.weight_context_simclr_loss[1] > 0.):
            n_org_seq = int(self.b_seq / 2)
            n_org_seg = int(n_org_seq * self.b_seg)
            emb_org = tf.reshape(emb_z[:n_org_seq,:,:], (n_org_seg, d))
            emb_rep = tf.reshape(emb_z[n_org_seq:,:,:], (n_org_seg, d)) # n_org_seg=n_rep_seg
            
            simclr_loss, _, _ = self.simclr_loss_func.batch_NTxent_loss(emb_org, emb_rep)
        else:
            simclr_loss = 0.      
    
        return loss * self.weight_context_simclr_loss[0], simclr_loss * self.weight_context_simclr_loss[1]# context_loss, simclr_loss

        
        
# Unit-test
# def test_ntxent_loss():
#     feat_dim = 128
#     n_org, n_rep = 3, 6
#     emb_org = tf.random.uniform((n_org, feat_dim))
#     emb_rep = tf.random.uniform((n_rep, feat_dim))
