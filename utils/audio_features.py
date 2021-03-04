"""
audio_features.py

USAGE:
    af = AudioFeat();
    audio, sr = af.load_audio('./hello.wav');
    melspec = af.melspectrogram(audio);
    mfcc_feat = af.mfcc(audio);
    
    af.display(melspec)
    
Created by Cochlear.ai, Last update: 12 Jun 2019 
"""
#%%
import tensorflow as tf # tensorflow>=2.0
import numpy as np
import warnings


class AudioFeat:
    def __init__(self):
        self._win_dict = {'hann': tf.signal.hann_window}
      
          
        
    def load_audio(self, fpath=str()):
        """
        return: (x, sr)
            x: (1xT) tensor audio samples
            sr: (int)
        """
        x, sr = tf.audio.decode_wav(tf.io.read_file(fpath))
        return tf.transpose(x), int(sr) # x: 1xT
    
    
    
    def spectrogram(self, x, nfft=1024, hop=230, pwr=2.0, win='hann'):
        """
        return: (FxTx1) power-spectrogram
        """   
        s = tf.signal.stft(x, nfft, hop, window_fn=self._win_dict[win],
                           pad_end=True)
        s = tf.abs(s)
        if pwr != 2.0:
            s = tf.pow(tf.sqrt(s), pwr)
        return tf.transpose(s, perm=[2,1,0])
    
    
    
    def melspectrogram(self,
                       x,
                       nfft=1024, # in Samples
                       hop=230,   # in Samples
                       sr=22050,
                       nmels=128,
                       fmin=0,    # Hz
                       fmax=None,  # Hz
                       return_db_melgram=True,
                       fweight='slaney',
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
        
        if fweight=='slaney':
            lin2mel_map = self._mel(sr, nfft, nmels, fmin, fmax, htk=reduce_dc) # (nbins x nmels)
        else:
            lin2mel_map = tf.signal.linear_to_mel_weight_matrix(nmels, nbins, sr,
                                                                fmin, fmax)
            
        s = tf.tensordot(tf.transpose(s, perm=[2,1,0]), lin2mel_map, axes=1) # (1xTxF)
        s = tf.transpose(s, perm=[2,1,0]) # (FxTx1)
        
        if return_db_melgram:
            s = 20 * tf.math.log(tf.maximum(s, 1e-10)) / tf.math.log(10.)
            s = s - tf.reduce_max(s)
            s = tf.math.maximum(s, -80.) # dynamic range = 80  
        return s
    
    
    
    def mfcc(self, x, nfft=1024, hop=230, nmfcc=13):
        """mel-frequency coefficients
        Returns
        -------
        mfcc   : (FxTx1)
        """
        
        s = self.melspectrogram(x, nfft=nfft, hop=hop) # (FxTx1)
        s = tf.transpose(s, [2,1,0]) # (1xTxF)
        s = tf.signal.mfccs_from_log_mel_spectrograms(s)[:, :, 1:nmfcc]
        return tf.transpose(s, [2,1,0])

    
    
    def display_spec(self, s, fi=list()):
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
    
    
    
    #-librosa-like mel-filter implementation-----------------------------------
    def _hz_to_mel(self, frequencies, htk=False):
        """Convert Hz to Mels
    
        """
        frequencies = np.asanyarray(frequencies)
    
        if htk:
            return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    
        # Fill in the linear part
        f_min = 0.0
        f_sp = 200.0 / 3
        mels = (frequencies - f_min) / f_sp
    
        # Fill in the log-scale part
        min_log_hz = 1000.0                         # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region
    
        if frequencies.ndim:
            # If we have array data, vectorize
            log_t = (frequencies >= min_log_hz)
            mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
        elif frequencies >= min_log_hz:
            # If we have scalar data, heck directly
            mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    
        return mels



    def _mel_to_hz(self, mels, htk=False):
        """Convert mel bin numbers to frequencies
        """
        mels = np.asanyarray(mels)
    
        if htk:
            return 700.0 * (10.0**(mels / 2595.0) - 1.0)
    
        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels
    
        # And now the nonlinear scale
        min_log_hz = 1000.0                         # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region
    
        if mels.ndim:
            # If we have vector data, vectorize
            log_t = (mels >= min_log_mel)
            freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
        elif mels >= min_log_mel:
            # If we have scalar data, check directly
            freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    
        return freqs



    def _fft_frequencies(self, sr=22050, n_fft=2048):
        '''Alternative implementation of `np.fft.fftfreq`
        '''
        return np.linspace(0, float(sr) / 2, int(1 + n_fft//2), endpoint=True)
    
    
    def _mel_frequencies(self, n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
        """Compute an array of acoustic frequencies tuned to the mel scale.
        .."https://librosa.github.io/librosa/_modules/librosa/core/time_frequency"
        """
        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = self._hz_to_mel(fmin, htk=htk)
        max_mel = self._hz_to_mel(fmax, htk=htk)
        mels = np.linspace(min_mel, max_mel, n_mels)
        return self._mel_to_hz(mels, htk=htk)



    def _mel(self, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
            norm=1):
        """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
        .. "https://librosa.github.io/librosa/_modules/librosa/filters.html#mel"
        """
        if fmax is None:
            fmax = float(sr) / 2
    
        if norm is not None and norm != 1 and norm != np.inf:
            raise NotImplementedError('Unsupported norm: {}'.format(repr(norm)))
    
        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)))
    
        # Center freqs of each FFT bin
        fftfreqs = self._fft_frequencies(sr=sr, n_fft=n_fft)
    
        # 'Center freqs' of mel bands - uniformly spaced between limits
        mel_f = self._mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)
    
        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i+2] / fdiff[i+1]
    
            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))
    
        if norm == 1:
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]
    
        # Only check weights if f_mel[0] is positive
        if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
            # This means we have an empty channel somewhere
            warnings.warn('Empty filters detected in mel frequency basis. '
                          'Some channels will produce empty responses. '
                          'Try increasing your sampling rate (and fmax) or '
                          'reducing n_mels.')
        return tf.constant(weights.transpose(1,0), dtype=tf.float32) 

        
    
#%%
def test1():
    af = AudioFeat();
    audio, sr = af.load_audio('../dataset/fma_small/000002.wav')
    melspec = af.melspectrogram(audio)
    mfcc_feat = af.mfcc(audio)
    return melspec








