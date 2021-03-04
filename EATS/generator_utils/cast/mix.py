import os
import librosa
import numpy as np
import scipy
import subprocess
import platform
import sys
from .core import *
#import matplotlib.pyplot as plt
import scipy.signal as signal 

# HPF
sos = signal.butter(10, 60, 'hp', fs=8000, output='sos')

def mixup(x1, x2, mixing_weight=None, alpha=1.0, mode='energy'):
  '''
    mixup(x1, x2, mixing_weight=None, alpha=1.0, mode='energy')
      Function that mixes two input audio signals with random ratio.

    arguments
      x1: Input audio signal.
      x2: Input audio signal.
      mixing_weight (optional): If the mixing weights for x1 (=w1) and x2 (=w2) are manually provided,
                                we use this instead of drawing random values.
      alpha (optional): The parameter tha controls the distribution of mixing weights.
                        alpha = 1 (default) makes the probability distribution of 
                        the mixing weights uniform in the range [0,1).
                        alpha > 1 causes the mixing weights to concentrate around 0.5.
                        alpha < 1 causes the mixing weights to be close to extremas (0, 1).
                        Note that the sum of mixing weights for x1 and x2 is always 1.
                        i.e., w1+w2=1
      mode (optional): The audio signals x1 and x2 are normalized ahead of the mixing.
                       This argument indicates the criterion for the normalization.
                       The default setting for this parameter is 'energy'.

    return
      x: Mix-up result. Note that the x is max-normalized before it is returned.
         If you want to change the amplitude, use amplitude_scaling() right after this.
      w1: Mixing weight for x1.
      w2: Mixing weight for x2.

    examples
      1. Default
        >> x, w1, w2 = mixup(x1, x2)
        >> x, w1, w2 = mixup(x1, x2, mode='energy')
      The mixing weights are randomly selected in the range [0.1, 0.9).
      Energy equalization of x1 and x2 -> linear combination (default).
   
      2. When the mixing weight is given.
        >> x, w1, w2 = mixup(x1, x2, mixing_weight=(0.3, 0.7))
        >> x, w1, w2 = mixup(x1, x2, mode='energy', mixing_weight=(0.3, 0.7))
      The mixing weights are 0.3 and 0.7 for x1 and x2, respectively.
      Energy equalization of x1 and x2 -> linear combination.
   
      3. Max value-based mixing
        >> x, w1, w2 = mixup(x1, x2, mode='max')
      The mixing weights are randomly selected in the range [0.1, 0.9).
      Max-normalization of x1 and x2 -> linear combination
   
      4. 2 & 3 combined
        >> x, w1, w2 = mixup(x1, x2, mode='max', mixing_weight=(0.3, 0.7))
      The mixing weights are 0.3 and 0.7 for x1 and x2, respectively.
      Max-normalization of x1 and x2 -> linear combination
   
      5. when alpha is given
        >> x, w1, w2 = mixup(x1, x2, alpha=0.2)
        >> x, w1, w2 = mixup(x1, x2, mode='energy', alpha=0.1)
      The mixing weights are more likely to be selected close to 
      either 0 or 1 compared to the default settings.
      Energy equalization of x1 and x2 -> linear combination.
   
      6. 3 & 5 combined
        >> x, w1, w2 = mixup(x1, x2, mode='max', alpha=20.0)
      The mixing weights are more likely to be selected in [0.4,0.6) 
      compared to the default settings.
      Max-normalization of x1 and x2 -> linear combination.
  '''

  # # # How to mix: define mixing_weights
  if mixing_weight is None: # randomly pick one in the range
    alpha = float(alpha)
    # Error
    if alpha <= 0:
      print('Error! Alpha value must be positive!')
      return 0

    # Mixing weight
    w1 = np.random.beta(alpha,alpha)
    w2 = 1-w1
  else:
    # Mixing weights are given
    mixing_weight = normalise(np.asarray(mixing_weight).astype(float))
    w1 = mixing_weight[0]
    w2 = mixing_weight[1]

    # Error
    if w1 > 1 or w2 > 1 or w1<=0 or w2<=0:
      print('Error! Weights not supported!')
      return 0


  # # # Length equalization
  L = np.max((len(x1),len(x2)))
  l = np.min((len(x1),len(x2)))
  x_tmp = np.zeros((L))
  start_idx = int((L-l)/2)
  if len(x1) < len(x2):
    x_tmp[start_idx:start_idx+l] += x1
    x1 = x_tmp
  elif len(x1) > len(x2):
    x_tmp[start_idx:start_idx+l] += x2
    x2 = x_tmp


  # # # Two different criteria: energy and max
  if mode == 'max':
    x = w1*max_normalise(x1)+w2*max_normalise(x2)
    x = max_normalise(x)

  elif mode == 'energy':
    E_x1 = energy_measure(x1)/float(len(x1)) # Mean energy per a sample
    x1 = x1/np.sqrt(E_x1)*w1

    E_x2 = energy_measure(x2)/float(len(x2))
    x2 = x2/np.sqrt(E_x2)*w2

    x = max_normalise(x1+x2)

  else:
    print('Wrong mix option! : '+mode)
    print('Switching to basic option (energy)')
    E_x1 = energy_measure(x1)/float(len(x1)) # Mean energy per a sample
    x1 = x1/np.sqrt(E_x1)*w1

    E_x2 = energy_measure(x2)/float(len(x2))
    x2 = x2/np.sqrt(E_x2)*w2

    x = max_normalise(x1+x2)

  return (x, w1, w2)


def background_mix(x, fs, x_bg=None, bg_folder='./bg', snr=None, snr_range=(-6, 6), unit='db', mode='energy'):
  '''
    background_mix(x, fs, bg_folder='./bg', snr=None, snr_range=(-6, 6), unit='db', mode='energy')
      Function that mixes the input audio signal with the background noise selected randomly 
      among the '.npy' files in the 'bg_folder'.
      The position of x is also randomly determined and is returned as an output.

    arguments
      x: Input audio signal (sound event).
      fs: Sample rate of x.
      x_bg (optional): Background noise signal. If not defined, this function randomly selects an audio file
                       among those in 'bg_folder'.
      bg_folder (optional): Folder that contains background sounds in the form of '.npy'.
                            The default setting is './bg'.
      snr (optional): If the desirable SNR is manually provided, 
                      we use this instead of drawing random values.
                      Here, 'S' is the power of sound event (x) 
                      and 'N' is the power of background sound.
      snr_range (optional): The range of the SNR for the event signal x.
                            If not provided, the default range is set to [-6dB, 6dB).
      unit (optional): Unit of 'snr' or 'snr_range'. Has to be either 'db' or 'linear'.
      mode (optional): The audio signals x and the randomly-selected background sound are normalized
                       ahead of the mixing. This argument indicates the criterion for the normalization.

    return
      x_mix: Background-mixed audio signal.
      SNR: dB-scale SNR
      Onset: Onset of the event in seconds
      Offset: Offset of the event in seconds

    examples
      1. Default
        >> x_mix, snr_db, onset, offset = background_mix(x, fs)
      Select an '.npy' file in './bg' folder, and mix the audio event x with the background sound.
      In mixing the audio, we first energy-based normalize on both background and event.
      Then we pick a random snr value in the range [-6, 6) presented in dB scale.

      2. Mix the event in the given SNR.
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, snr=3)
      SNR of the event in the result file is 3dB.

      3. If you want to newly provide the SNR range.
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, snr_range=(-3,3))
      The lower bound and the upper bound is switched to -3dB and 3dB, respectively.

      4. When the SNR scale is not dB, but linear.
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, snr=1.5, unit='linear')
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, snr_range=(0.2, 1.5), unit='linear')

      5. Normalization method: 'energy' (default) and 'max'
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, mode='max')

      6. If you want to change the path of background audio files.
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, bg_folder='./path_to_background')

      7. If you have a specific background noise that you want to mix with the signal
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, x_bg)
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, x_bg, snr_range=(-3,3), unit='db')
        >> x_mix, snr_db, onset, offset = background_mix(x, fs, x_bg, snr=3)
  '''
  # You are encouraged to run 'audio_to_npy' ahead of this function.
  
  # If background noise signal is not defined, we randomly pick one in the 'bg_folder'.
  if type(x_bg) is type(None):
    # Load one of the bg
    if sys.version_info[0] == 2:
      (loc, _, filenames) = os.walk(bg_folder).next()
    elif sys.version_info[0] == 3:
      (loc, _, filenames) = os.walk(bg_folder).__next__()
    else:
      print('Unsupported python version.')
      (loc, _, filenames) = os.walk(bg_folder).__next__()
    filenames = [f for f in filenames if not f.startswith('.')]
  
    while True:
      bg_idx = np.random.randint(len(filenames))

      if filenames[bg_idx].endswith('.npy'):
        x_bg = np.load(bg_folder+'/'+filenames[bg_idx])
        if energy_measure(x_bg) != 0:
          break
      # elif filenames[bg_idx].endswith('.wav') or filenames[bg_idx].endswith('.mp3'):
      #   x_bg, fs_bg = librosa.core.load(bg_folder+'/'+filenames[bg_idx])
      else:
        pass
  else:
    pass  # We already have a background noise.
  
  # # # How to mix: define snr
  if snr is None: # randomly pick one in the range
    snr_range = np.sort(snr_range)
    min_snr = snr_range[0]
    max_snr = snr_range[1]

    snr = random_number(range=(min_snr, max_snr))
  else:
    pass  # No problem

  # # # Mix
  # 0. If event length is longer than the background,
  # we cut the event.
  if len(x) > len(x_bg):
    x = x[:len(x_bg)]

  # 1. event position
  while True:
    start_idx = int(random_number(range=(0, len(x_bg)-len(x))))
    end_idx = start_idx+len(x)
    if energy_measure(x_bg[start_idx:end_idx]) != 0:
      break
  x_bg = x_bg[start_idx:end_idx]

  # HP filter
  #x_bg = signal.sosfilt(sos, x_bg)
  

  # 2. (energy/max) normalize
  if mode == 'max':
    x_bg = x_bg/np.max(np.abs(x_bg))
    x = max_normalise(x)
  elif mode == 'energy':
    E_bg = energy_measure(x_bg/100)
    x_bg = x_bg/np.sqrt(E_bg)
    E_x = energy_measure(x/100)
    x = x/np.sqrt(E_x)
    #print('x: {}'.format(np.sum(x*x)))
    #print('x_bg: {}'.format(np.sum(x_bg*x_bg)))
  else:
    print('Error! Wrong background mix option!: '+str(mode))
    print('Switching to basic option (energy)')
    E_bg = energy_measure(x_bg)/float(len(x_bg))
    x_bg = x_bg/np.sqrt(E_bg)
    E_x = energy_measure(x)/float(len(x))
    x = x/np.sqrt(E_x)

  # 3. snr unit
  if unit is 'db' or unit is 'dB' or unit is 'DB':
    magnitude = np.power(10,snr/20.0) # This includes sqrt.
    snr_db = snr
  elif unit is 'linear' or unit is 'Linear':
    magnitude = np.sqrt(snr)  # Because SNR is basically a power ratio.
    snr_db = 10.0*np.log10(snr)
  else:
    print('Error! Invalid SNR unit: '+str(unit))
    print('Switching to basic option (dB)')
    magnitude = np.power(10,snr/20.0) # This includes sqrt.
    snr_db = snr
#
#  plt.figure()
#  plt.plot(x)
#  plt.figure()
#  plt.plot(x_bg)
#  plt.show()
#  print('x: {}'.format(np.sum(x*x)))
#  print('x_bg: {}'.format(np.sum(x_bg*x_bg)))
#  input('.....')


  # 4. mix!
  x_mix = x_bg + magnitude*x
#  x_mix = x_bg + magnitude * x
#  print('m=',magnitude)
#  print('max_bg=', np.max(x_bg))
#  print('max_x=', np.max(x))
  x_mix = max_normalise(x_mix)

  onset = start_idx/float(fs)
  offset = end_idx/float(fs)

  return (x_mix, snr_db, onset, offset)







