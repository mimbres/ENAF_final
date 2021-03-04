import tensorflow as tf
from tensorflow.keras.utils import Sequence
from .generator_utils.utils import bg_mix_batch, ir_aug_batch, load_audio, get_fns_seg_list, calculate_offset_padoffset
from .generator_utils.utils import load_audio_multi_start
import numpy as np
"""
OPTIONS:
    - offset_margin_hop_rate: this activates only if random_offset=True. 0.4 was default in Unit A.
    - speech_mix_parameter: [False] or [True, speech_tr_fps, (SNRmin,SNRmax)]

"""
MAX_IR_LENGTH = 400 # 50ms with fs=8000


class genUnbalSequence(Sequence) : 
    def __init__(self,
                 fns_event_list=None,
                 bsz=120, # If being used with TPUs, bsz means global batch size
                 n_anchor=60, #ex) bsz=40, n_anchor=8 : 4 positive samples for 1 anchor (with TPUs, this means global n_anchor)
                 duration=1, # duration in seconds
                 hop=.5,
                 fs=8000,
                 shuffle=False,
                 seg_mode="all",
                 amp_mode='normal',
                 random_offset=False,
                 offset_margin_hop_rate=0.4, # this activates only if random_offset=True
                 bg_mix_parameter=[False],
                 ir_mix_parameter=[False],
                 speech_mix_parameter=[False],
                 reduce_items_to=0, # for debugging with small number of examples
                 test_mode=False):
        
        self.bsz = bsz
        self.n_anchor = n_anchor
        # Deciding size of batch, anchor, and positive samples
        if bsz!=n_anchor:
            self.n_pos_per_anchor = round((bsz - n_anchor) / n_anchor)
            self.n_pos_bsz = bsz - n_anchor
        else:
            self.n_pos_per_anchor = 0
            self.n_pos_bsz = 0
                    
        self.duration = duration
        self.hop = hop
        self.fs = fs
        self.shuffle = shuffle        
        self.seg_mode = seg_mode 
        self.amp_mode = amp_mode
        self.random_offset = random_offset
        self.offset_margin_hop_rate = offset_margin_hop_rate
        self.offset_margin_frame = int(hop * self.offset_margin_hop_rate * fs)
        
        self.bg_mix = bg_mix_parameter[0]
        self.ir_mix = ir_mix_parameter[0]
        self.speech_mix = speech_mix_parameter[0]

        if self.bg_mix == True:
            fns_bg_list = bg_mix_parameter[1]
            self.bg_snr_range = bg_mix_parameter[2]
            
        if self.ir_mix == True:
            fns_ir_list = ir_mix_parameter[1]
        
        if self.speech_mix == True:
            fns_speech_list = speech_mix_parameter[1]
            self.speech_snr_range = speech_mix_parameter[2]
        
        if self.seg_mode in {'random_oneshot', 'all'}:
            self.fns_event_seg_list = get_fns_seg_list(fns_event_list,
                                                       self.seg_mode,
                                                       self.fs,
                                                       self.duration,
                                                       hop=self.hop)
            # [[filename, seg_idx, offset_min, offset_max], [ ... ] , ... [ ... ]]
        else:
            raise NotImplementedError("seg_mode={}".format(self.seg_mode))
        
        self.n_samples = int((len(self.fns_event_seg_list) // n_anchor) * n_anchor)
        
        if self.shuffle == True:
            self.index_event = np.random.permutation(self.n_samples)
        else:
            self.index_event = np.arange(self.n_samples)

        if self.bg_mix == True:
            self.fns_bg_seg_list = get_fns_seg_list(fns_bg_list, 'all', self.fs, self.duration)
            self.n_bg_samples = len(self.fns_bg_seg_list)
            if self.shuffle == True:
                self.index_bg = np.random.permutation(self.n_bg_samples)
            else:
                self.index_bg = np.arange(self.n_bg_samples)
        else:
            pass

        if self.speech_mix == True: 
            self.fns_speech_seg_list = get_fns_seg_list(fns_speech_list, 'all', self.fs, self.duration)
            self.n_speech_samples = len(self.fns_speech_seg_list)
            if self.shuffle == True:
                self.index_speech = np.random.permutation(self.n_speech_samples)
            else:
                self.index_speech = np.arange(self.n_speech_samples)
            
            
        if self.ir_mix == True:
            self.fns_ir_seg_list = get_fns_seg_list(fns_ir_list, 'first', self.fs, self.duration)
            self.n_ir_samples = len(self.fns_ir_seg_list)
            if self.shuffle == True:
                self.index_ir = np.random.permutation(self.n_ir_samples)
            else:
                self.index_ir = np.arange(self.n_ir_samples)
        else:
            pass
        
        self.reduce_items_to = reduce_items_to
        self.test_mode = test_mode
        if test_mode:
            self.test_mode_offset_sec_list = ((np.arange(self.n_pos_per_anchor) - (self.n_pos_per_anchor-1)/2) / self.n_pos_per_anchor) * self.hop

        

    def __len__(self) : 
        # Denotes the number of batches per epoch
        if self.reduce_items_to != 0:
            return int(np.ceil(self.n_samples / float(self.n_anchor)) * self.reduce_items_to)
        else:
            return int(np.ceil(self.n_samples / float(self.n_anchor)))


    def on_epoch_end(self) : 
        if self.shuffle == True:
            self.index_event = list(np.random.permutation(self.n_samples))
        else:
            pass

        if self.bg_mix == True and self.shuffle == True:   
            self.index_bg = list(np.random.permutation(self.n_bg_samples)) # same number with event samples
        else:
            pass
        
        if self.ir_mix == True and self.shuffle == True:   
            self.index_ir = list(np.random.permutation(self.n_ir_samples)) # same number with event samples
        else:
            pass

        if self.speech_mix == True and self.shuffle == True:   
            self.index_speech = list(np.random.permutation(self.n_speech_samples)) # same number with event samples
        else:
            pass        

    def __getitem__(self, idx) :
        # Get anchor and positive samples
        index_anchor_for_batch = self.index_event[idx*self.n_anchor: (idx+1)*self.n_anchor]
        Xa_batch, Xp_batch = self.__event_batch_load(index_anchor_for_batch)
        global bg_sel_indices, speech_sel_indices
        
        if self.bg_mix and self.speech_mix: 
            if self.n_pos_bsz > 0:
                # Prepare bg for positive samples
                bg_sel_indices = np.arange(idx * self.n_pos_bsz,
                                            (idx+1) * self.n_pos_bsz) % self.n_bg_samples
                index_bg_for_batch = self.index_bg[bg_sel_indices]
                Xp_bg_batch = self.__bg_batch_load(index_bg_for_batch)

                # Prepare speech for positive samples
                speech_sel_indices = np.arange(idx * self.n_pos_bsz,
                                            (idx+1) * self.n_pos_bsz) % self.n_speech_samples
                index_speech_for_batch = self.index_speech[speech_sel_indices]
                Xp_speech_batch = self.__speech_batch_load(index_speech_for_batch)

                Xp_noise_batch = Xp_bg_batch + Xp_speech_batch
                # mix
                Xp_batch = bg_mix_batch(Xp_batch, Xp_noise_batch, self.fs, snr_range=self.speech_snr_range) 

            
        else:
            if self.bg_mix == True:
                if self.n_pos_bsz > 0:
                    # Prepare bg for positive samples
                    bg_sel_indices = np.arange(idx * self.n_pos_bsz,
                                                (idx+1) * self.n_pos_bsz) % self.n_bg_samples
                    index_bg_for_batch = self.index_bg[bg_sel_indices]
                    Xp_bg_batch = self.__bg_batch_load(index_bg_for_batch)
                    # mix
                    Xp_batch = bg_mix_batch(Xp_batch, Xp_bg_batch, self.fs, snr_range=self.bg_snr_range) 
            else:
                pass
            
            if self.speech_mix == True:
                if self.n_pos_bsz > 0:
                    # Prepare speech for positive samples
                    speech_sel_indices = np.arange(idx * self.n_pos_bsz,
                                                (idx+1) * self.n_pos_bsz) % self.n_speech_samples
                    index_speech_for_batch = self.index_speech[speech_sel_indices]
                    Xp_speech_batch = self.__speech_batch_load(index_speech_for_batch)
                    # mix
                    Xp_batch = bg_mix_batch(Xp_batch, Xp_speech_batch, self.fs, snr_range=self.bg_snr_range) 
            else:
                pass

            
            
        if self.ir_mix == True:
            if self.n_pos_bsz > 0:
                # Prepare ir for positive samples
                ir_sel_indices = np.arange(idx * self.n_pos_bsz,
                                           (idx+1) * self.n_pos_bsz) % self.n_ir_samples
                index_ir_for_batch = self.index_ir[ir_sel_indices]
                Xp_ir_batch = self.__ir_batch_load(index_ir_for_batch)
                
                # ir aug
                Xp_batch = ir_aug_batch(Xp_batch, Xp_ir_batch)
        else:
            pass    
        
        Xa_batch = np.expand_dims(Xa_batch, 1).astype(np.float32) # (n_anchor, 1, T)
        Xp_batch = np.expand_dims(Xp_batch, 1).astype(np.float32) # (n_pos, 1, T)
        return Xa_batch, Xp_batch
        


    def __event_batch_load(self, anchor_idx_list):
        # Get Xa_batch and Xp_batch for anchor and positive samples
        Xa_batch = None
        Xp_batch = None
        for idx in anchor_idx_list: # idx: index for one sample
            pos_start_sec_list = []
            # fns_event_seg_list = [[filename, seg_idx, offset_min, offset_max], [ ... ] , ... [ ... ]]
            offset_min, offset_max = self.fns_event_seg_list[idx][2], self.fns_event_seg_list[idx][3] 
            anchor_offset_min = np.max([offset_min, -self.offset_margin_frame])
            anchor_offset_max = np.min([offset_max, self.offset_margin_frame]) 
            if (self.random_offset==True) & (self.test_mode==False):
                np.random.seed(idx) 
                # Calculate anchor_start_sec
                _anchor_offset_frame = np.random.randint(low=anchor_offset_min,
                                                     high=anchor_offset_max)
                _anchor_offset_sec = _anchor_offset_frame / self.fs
                anchor_start_sec = self.fns_event_seg_list[idx][1] * self.hop + _anchor_offset_sec
            else:
                _anchor_offset_frame = 0
                anchor_start_sec = self.fns_event_seg_list[idx][1] * self.hop

                             
                
            # Calculate multiple(=self.n_pos_per_anchor) pos_start_sec
            if self.n_pos_per_anchor > 0:
                pos_offset_min = np.max([(_anchor_offset_frame - self.offset_margin_frame), offset_min])
                pos_offset_max = np.min([(_anchor_offset_frame + self.offset_margin_frame), offset_max])
                if (self.test_mode==False):  
                    _pos_offset_frame_list = np.random.randint(low=pos_offset_min,
                                                               high=pos_offset_max,
                                                               size=self.n_pos_per_anchor)
                    _pos_offset_sec_list = _pos_offset_frame_list / self.fs
                    pos_start_sec_list = self.fns_event_seg_list[idx][1] * self.hop + _pos_offset_sec_list
                else: # In test_mode, we fix offset by nth positive copy
                    _pos_offset_sec_list = self.test_mode_offset_sec_list # [-0.2, -0.1,  0. ,  0.1,  0.2] for n_pos=5 with hop=0.5s
                    _pos_offset_sec_list[(_pos_offset_sec_list < pos_offset_min / self.fs)] = pos_offset_min / self.fs
                    _pos_offset_sec_list[(_pos_offset_sec_list > pos_offset_max / self.fs)] = pos_offset_max / self.fs
                    pos_start_sec_list = self.fns_event_seg_list[idx][1] * self.hop + _pos_offset_sec_list

            
            # load audio: anchor, pos1, pos2,..pos_n
            #print(self.fns_event_seg_list[idx]) #
            start_sec_list = np.concatenate(([anchor_start_sec], pos_start_sec_list))            
            xs = load_audio_multi_start(self.fns_event_seg_list[idx][0],
                                        start_sec_list,
                                        self.duration,
                                        self.fs,
                                        self.amp_mode) # xs: ((1+n_pos)),T)
            # Check NaN
            assert np.any(np.isnan(xs))==False, idx
            
            
            if Xa_batch is None:
                Xa_batch = xs[0,:].reshape((1,-1))
                Xp_batch = xs[1:,:] # If self.n_pos_per_anchor==0: this produces an empty array
            else:
                Xa_batch = np.vstack((Xa_batch, xs[0,:].reshape((1,-1))))  # Xa_batch: (n_anchor, T)
                Xp_batch = np.vstack((Xp_batch, xs[1:,:])) # Xp_batch: (n_pos, T)
        return Xa_batch, Xp_batch



    def __bg_batch_load(self, idx_list):
        X_bg_batch = None # (n_batch+n_batch//n_class, fs*k)
        random_offset_sec = np.random.randint(0, self.duration * self.fs/2,
                                              size=len(idx_list)) / self.fs 
        for i, idx in enumerate(idx_list):
            idx = idx % self.n_bg_samples
            offset_sec = np.min([random_offset_sec[i], 
                                 self.fns_bg_seg_list[idx][3] / self.fs])

            X = load_audio(filename=self.fns_bg_seg_list[idx][0],
                           seg_start_sec=self.fns_bg_seg_list[idx][1] * self.duration,
                           offset_sec=offset_sec,
                           seg_length_sec=self.duration,
                           seg_pad_offset_sec=0.,
                           fs=self.fs,
                           amp_mode='normal')

            X = X.reshape(1,-1)

            if X_bg_batch is None : 
                X_bg_batch = X
            else :
                X_bg_batch = np.concatenate((X_bg_batch, X), axis=0)

        return X_bg_batch

    def __speech_batch_load(self, idx_list):
        X_speech_batch = None # (n_batch+n_batch//n_class, fs*k)
        random_offset_sec = np.random.randint(0, self.duration * self.fs/2,
                                              size=len(idx_list)) / self.fs 
        for i, idx in enumerate(idx_list):
            idx = idx % self.n_speech_samples
            offset_sec = np.min([random_offset_sec[i], 
                                 self.fns_speech_seg_list[idx][3] / self.fs])

            X = load_audio(filename=self.fns_speech_seg_list[idx][0],
                           seg_start_sec=self.fns_speech_seg_list[idx][1] * self.duration,
                           offset_sec=offset_sec,
                           seg_length_sec=self.duration,
                           seg_pad_offset_sec=0.,
                           fs=self.fs,
                           amp_mode='normal')

            X = X.reshape(1,-1)

            if X_speech_batch is None : 
                X_speech_batch = X
            else :
                X_speech_batch = np.concatenate((X_speech_batch, X), axis=0)

        return X_speech_batch
    
    
    def __ir_batch_load(self, idx_list):
        X_ir_batch = None # (n_batch+n_batch//n_class, fs*k)

        for idx in idx_list:
            idx = idx % self.n_ir_samples

            X = load_audio(filename=self.fns_ir_seg_list[idx][0],
                           seg_start_sec=self.fns_ir_seg_list[idx][1] * self.duration,
                           offset_sec=0.0,
                           seg_length_sec=self.duration,
                           seg_pad_offset_sec=0.0,
                           fs=self.fs,
                           amp_mode='normal')
            # 2020. 10. 08, limit max IR length..
            # max_ir_length = MAX_IR_LENGTH + np.random.randint(-200, 200) # randomness
            # max_ir_length = min(max_ir_length, len(X)) # avoid exceeding original IR length
            if len(X) > MAX_IR_LENGTH:
                X = X[:MAX_IR_LENGTH]
            
            X = X.reshape(1,-1)

            if X_ir_batch is None : 
                X_ir_batch = X
            else :
                X_ir_batch = np.concatenate((X_ir_batch, X), axis=0)

        return X_ir_batch