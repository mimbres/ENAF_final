from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
import os
import librosa
import numpy as np
import scipy
from .core import *
import sys


def _draw_seed(sd=0.5, min=-3, max=3):
  '''
    _draw_seed
      Pick a random value in semitones which would be used for time_stretch/pitch_up/speed_up.

    arguments
      sd: Standard deviation of the Gaussian distribution used to pick the random seed.
      min: Minimum value (lower bound) of the output.
      max: Maximum value (upper bound) of the output.

    return
      Gaussian-randomly picked value. If the picked value exceeds [min, max], it is clipped to min or max.
  '''

  # Gaussian distribution
  aug_seed = np.random.randn()*sd

  # Clipping to make aug_seed ~ [-3,3]
  aug_seed = np.maximum(aug_seed,min)
  aug_seed = np.minimum(aug_seed,max)

  return aug_seed


def time_stretch(x, fs, r=1.0):
  '''
    time_stretch
      Function for time stretch augmentation of the input audio.
      If r > 1, this function lengthens the input audio.
      Else if r < 1, the audio shrinks.

    arguments
      x: Input audio signal.
      fs: Sample rate.
      r: How much to enlength x (in linear scale, not in semitones)

    return
      Time-stretched audio signal.
  '''

  # save -> audiotsm library's phase vocoder -> load
  scipy.io.wavfile.write('temporary.wav',fs,np.int16(32768*x))
  with WavReader('temporary.wav') as reader:
    with WavWriter('temporary_aug.wav', reader.channels, reader.samplerate) as writer:
      tsm = phasevocoder(reader.channels, speed=(1/r))  # Shrinks if r<1
      tsm.run(reader, writer)
  x_aug,fs = librosa.core.load('temporary_aug.wav',sr=fs)

  # delete the temporarily saved files
  os.remove('temporary_aug.wav')
  os.remove('temporary.wav')
  return x_aug


def pitch_up(x, fs, r=1.0):
  '''
    pitch_up(x, fs, r=1.0)
      Function for pitch shifting which consists of time_stretch followed by speed_up
      If r > 1, this function increases the pitch of the input audio.
      Else if r < 1, the pitch decreases.

    arguments
      x: Input audio signal.
      fs: Sample rate.
      r: How much to increase pitch (in linear scale, not in semitones)

    return
      Pitch-shifted audio signal.
  '''

  x_tmp = time_stretch(x,fs,r)
  x_aug = speed_up(x_tmp,fs,r) # Pitch up down if r>1
  return x_aug


def speed_up(x, fs, r=1.0):
  '''
    speed_up(x, fs, r=1.0)
      Function for speeding up (when r > 1) the input audio play speed.
      This in fact works by downsampling the input and play it with the original sample rate.

    arguments
      x: Input audio signal.
      fs: Sample rate.
      r: How much to speed up (in linear scale, not in semitones)

    return
      Play speed-augmented audio signal.
  '''

  x_aug = librosa.resample(x,fs,fs/r) # Shrinks if r>1 (mosquito sound...)
  return x_aug


def ir_aug(x, fs, ir_folder='./ir'):
  '''
    ir_aug(x, fs, ir_folder='./ir')
      Function that randomly picks an impulse response (IR) and performs convolution operation with the input audio.

    arguments
      x: Input audio signal.
      fs: Sample rate.
      ir_folder: The folder where the IRs are saved. 
                 You are encouraged to run 'audio_to_npy' function before this.

    return
      IR-augmented audio signal.
  '''

  # Here we assume that only '.npy' files exist in 'ir_folder'.
  # The '.npy' files are assumed to contain IRs resampled to 'fs' in advance.
  if sys.version_info[0] == 2:
    (loc, _, filenames) = os.walk(ir_folder).next()
  elif sys.version_info[0] == 3:
    (loc, _, filenames) = os.walk(ir_folder).__next__()
  else:
    print('Unsupported python version.')
    (loc, _, filenames) = os.walk(ir_folder).__next__()

  filenames = [f for f in filenames if not f.startswith('.')]

  # Randomly chosen IR index
  idx_ir = np.random.randint(low=0,high=len(filenames))

  # Load IR -> '.npy' file
  x_ir = np.load(ir_folder+'/'+filenames[idx_ir])

  # Length equalization
  if len(x) < len(x_ir):
    # If x_ir is longer than x, the long tail is considered useless.
    x_ir = x_ir[0:len(x)]
  elif len(x) > len(x_ir):
    # zero-padding
    x_ir = np.concatenate((x_ir,np.zeros(len(x)-len(x_ir))),axis=0)

  # FFT -> multiply -> IFFT
  fftLength = np.maximum(len(x),len(x_ir))
  X = np.fft.fft(x,n=fftLength)
  X_ir = np.fft.fft(x_ir,n=fftLength)
  x_aug = np.fft.ifft(np.multiply(X_ir,X))[0:len(x)].real
  x_aug = x_aug/np.max(np.abs(x_aug)) # Max-normalize
  return x_aug


def amplitude_scaling(x, scale_range=(0.1,1.0)):
  '''
    amplitude_scaling(x, scale_range=(0.1,1.0))
      Function that changes the amplitude of the input sound by multiplying a random value.

    arguments
      x: Input audio signal.
      scale_range (optional): The range of the value to multiply.
                              If not provided, default range is [0.1, 1.0).

    return
      Amplitude-rescaled audio signal.

    examples
      1. Default
        >> x = amplitude_scaling(x)
      x = SOME_VALUE_BETWEEN_0.1_AND_1.0 * x

      2. When the scale_range is manually given
        >> x = amplitude_scaling(x, scale_range=(0.2, 1.2))
      x = SOME_VALUE_BETWEEN_0.2_AND_1.2 * x
  '''

  # # # Define the value to multiply
  # min and max value of magnitude
  scale_range = np.sort(scale_range)
  magnitude = random_number(range=scale_range)

  return magnitude*x


def augmentation(x, fs, *args, **kwargs):
  '''
    augmentation(x, fs, ir_folder='./ir')
      Function that wraps various augmentation functions to provide friendly interface.

    arguments
      x: Input audio signal.
      fs: Sample rate.
      args (optional): Select one or more augmentation options 
                       among 'tempo', 'pitch', 'speed', 'ir', and 'amplitude'.
      kwargs (optional): shirt_ratio, ir_folder
        shift_ratio: How much to time-stretch, pitch-shift, speed-up.
                     This value is represented in the scale of semitone.
        ir_folder: The folder where the IRs are saved.
                   This is fed into ir_aug function directly.
                   You are encouraged to run 'audio_to_npy' function before this.

    return
      x: Augmented audio signal
      fs: Sampling rate.
  '''

  # Determine shift ratio if not defined yet
  if 'shift_ratio' in kwargs:
    shift_ratio = kwargs['shift_ratio']
  else:
    shift_ratio = _draw_seed() # Semi-tone scale
  # Semi-tone scale to linear scale
  shift_ratio = np.power(2,shift_ratio/float(12))

  # Determine ir_folder if not defined yet
  if 'ir_folder' in kwargs:
    ir_folder = kwargs['ir_folder']
  else:
    ir_folder = 'ir'

  if 'tempo' in args:
    x = time_stretch(x,fs,shift_ratio)

  if 'pitch' in args:
    x = pitch_up(x,fs,shift_ratio)

  if 'speed' in args:
    x = speed_up(x,fs,shift_ratio)

  if 'ir' in args:
    x = ir_aug(x,fs,ir_folder)

  if 'amplitude' in args:
    x = amplitude_scaling(x)

  return (x,fs)




