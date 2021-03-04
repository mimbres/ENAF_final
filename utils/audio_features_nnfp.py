"""
audio_features.py

USAGE:
    af = AudioFeat();
    audio, sr = af.load_audio('./utils/small_dataset/001040.mp3_8k.wav');
    melspec = af.melspectrogram(audio);
    mfcc_feat = af.mfcc(audio);
    
    af.display(melspec)
    
Created by Cochlear.ai, Last update: 8 Jul 2019 
"""
#%%
import tensorflow as tf # tensorflow>=2.0
assert tf.__version__ >= "2.0"


class AudioFeat:
    def __init__(self):
        self._win_dict = {'hann': tf.signal.hann_window}  
          
        
    
    # Audio loading with TF-native function     
    def load_audio(self, fpath=str()):
        """
        return: (x, sr)
            x: (1xT) tensor audio samples
            sr: (int)
        """
        x, sr = tf.audio.decode_wav(tf.io.read_file(fpath))
        return tf.transpose(x), sr # x: 1xT

    

    def spectrogram(self, x, nfft=1024, hop=230, pwr=2.0, win='hann',
                    frange=None,
                    return_db_gram=False, reduce_dc=False):
        """
        Arguements
        ----------
        - frange: None or [fmin, fmax, sr] 
            
        Return
        ------
        - (FxTx1) power-spectrogram
        """   
        if frange!=None:
            fmin, fmax, sr = frange
            hcut = (sr / 2) - fmax # High-cut bandwidth in Hz
            lcut = fmin # Low-cut bandwidth in Hz
            nbin_lcut = round(lcut /  (sr / nfft))
            nbin_hcut = round(hcut /  (sr / nfft))
            """
            nfft_ext = round(nfft*(hcut + lcut) / (sr - hcut - lcut))
            nfft += nfft_ext
            nbin_lcut = round(lcut /  (sr / nfft))
            nbin_hcut = nfft_ext - nbin_lcut # This code doesn't work, bcoz TF stft allows only 2**n nfft size.
            """
        s = tf.signal.stft(x, nfft, hop, window_fn=self._win_dict[win],
                           pad_end=True)   
        s = tf.abs(s)
        if pwr != 1.0:
            s = tf.pow(s, pwr)
        s = tf.transpose(s, perm=[2,1,0])
        
        if reduce_dc:
            s = s[1:,:,:]
        if frange!=None:
            if nbin_lcut!=0:
                zeros = tf.zeros_like(s[:nbin_lcut,:,:])
                s = tf.concat([zeros, s[nbin_lcut:,:,:]], axis=0)
            if nbin_hcut!=0:
                zeros = tf.zeros_like(s[-nbin_hcut:,:,:])
                s = tf.concat([s[:-nbin_hcut,:,:], zeros], axis=0)
                
        if return_db_gram:
            s = 10 * tf.math.log(tf.maximum(s, 1e-10)) / tf.math.log(10.)
            s = s - tf.reduce_max(s)
            s = tf.math.maximum(s, -80.) # dynamic range = 80  
        return s            
    
    
    
    def melspectrogram(self,
                       x,
                       nfft=1024, # in Samples
                       hop=230,   # in Samples
                       sr=22050,
                       nmels=256,
                       fmin=0,    # Hz
                       fmax=None,  # Hz
                       return_db_melgram=True,
                       fweight='flat',
                       reduce_dc=True, 
                       pwr=2. ):
        """Generate a Mel-spectrogram
        Arguments
        ---------
        x       : (1xT) mono audio samples 
        ...        
        fweight : use 'slaney' for librosa-like features
            
        Returns
        -------
        s       : (FxTx1)
        """
        s = self.spectrogram(x, nfft=nfft, hop=hop, pwr=pwr) # (FxTx1)
        
        nbins = s.shape[0]
        if fmax==None: fmax = sr/2
        
        lin2mel_map = tf.signal.linear_to_mel_weight_matrix(nmels, nbins, sr,
                                                            fmin, fmax)
            
        s = tf.tensordot(tf.transpose(s, perm=[2,1,0]), lin2mel_map, axes=1) # (1xTxF)
        s = tf.transpose(s, perm=[2,1,0]) # (FxTx1)
        
        if return_db_melgram:
            s = 10 * tf.math.log(tf.maximum(s, 1e-10)) / tf.math.log(10.)
            s = s - tf.reduce_max(s)
            s = tf.math.maximum(s, -80.) # dynamic range = 80  
        return s

        
    
#%%
def display_spec(s, fi=list()):
    """
    s : input spectrogram or mel-spectrogram, FxTx(1).
    fi: (optional) frequency index
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if type(s) is not np.ndarray:
        s = s.numpy()
    plt.imshow(s.squeeze(), origin='lower', aspect='auto', cmap='jet')
    return        
    
def test_spec():
    af = AudioFeat();
    #audio, sr = af.load_audio('utils/small_dataset/001039.mp3_8k.wav')
    audio, sr = af.load_audio('utils/small_dataset/001040.mp3_8k.wav')
    #audio, sr = af.load_audio('utils/small_dataset/classical.00099_8k.wav')
    audio, sr = af.load_audio('utils/small_dataset/pop.00088_8k.wav')
    spec = af.spectrogram(audio, nfft=512, hop=256, pwr=2., win='hann',
                          frange=[300,4000,8000], return_db_gram=True,
                          reduce_dc=True)
    display_spec(spec[:,:500,:])
    return


def test_melspec():
    af = AudioFeat();
    #audio, sr = af.load_audio('utils/small_dataset/001039.mp3_8k.wav')
    audio, sr = af.load_audio('utils/small_dataset/001040.mp3_8k.wav')
    #audio, sr = af.load_audio('utils/small_dataset/classical.00099_8k.wav')
    audio, sr = af.load_audio('utils/small_dataset/pop.00088_8k.wav')
    melspec = af.melspectrogram(audio, nfft=1024, hop=256, sr=int(sr), 
                                nmels=256, fmin=300, fmax=4000,
                                return_db_melgram=True, pwr=2.)
    display_spec(melspec[:,:500,:])
    return






