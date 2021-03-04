import os
import numpy as np
import scipy
from scipy.io.wavfile import write
import subprocess
import librosa
import platform
import sys


def normalise(x):
  '''
    normalise(x)
      Function to make sum(abs(x))=1.

    arguments
      x: Input audio signal.

    return
      Sum-normalized audio signal. x/np.sum(np.abs(x))
  '''
  
  if np.sum(np.abs(x)) == 0:
    return x
  else:
    return x/np.sum(np.abs(x))


def max_normalise(x):
  '''
    max_normalise(x)
      Function to max-normalize the input audio x.

    arguments
      x: Input audio signal.

    return
      Max-normalized audio signal. x/np.max(np.abs(x))
  '''
  
  if np.max(np.abs(x)) == 0:
    return x
  else:
    return x/np.max(np.abs(x))


def energy_measure(x):
  '''
    energy_measure(x)
      Function to measure signal energy.

    arguments
      x: Input audio signal.

    return
      np.sum(x*x)
  '''

  return np.sum(x*x)


def random_number(range=(0,1), random_seed=0):
  '''
    random_number(range=(0,1))
      Function to pick a real-number randomly in the input range.

    arguments
      range: Range of the output. If not provided, the default range is used: [0,1).
             range[0] is the lower-bound of the output.
             range[1] is the upper-bound of the output.

    return
      A real-number that is uniform-randomly picked in the range.
  '''

  range = np.sort(range)
  range1 = range[0]
  range2 = range[1]
  
  np.random.seed(random_seed)
  # Uniform random real-number
  return (range2-range1)*np.random.random((1))[0]+range1


def soundsc(x, fs, filename=None):
  '''
    soundsc(x, fs, filename=None)
      Function for playing 'max-normalized' audio signal to listen. Only supports mac and linux.

    arguments
      x: Audio signal.
      fs: Sample rate of the audio signal x.
      filename (optional): Name of the audio file to save. Has to end with '.wav'.
                           This is required only if you want to remain the audio file.

    return
      Nothing is returned but the 'normalized' sound is played.
  '''

  # The audio file remains only if 'filename' is given.
  save_flag = 1
  if filename is None:
    save_flag = 0
    filename = 'cast_tmp.wav'

  assert filename.endswith('.wav'), 'Only wav format is supported!'

  write(filename, fs, max_normalise(x))
  # now that file is written to the disk - play it
  if platform.system() == 'Linux':
    subprocess.call(["play", filename]) #<-  for linux
  elif platform.system() == 'Darwin':
    subprocess.call(["afplay", filename]) #<- for Mac
  else:
    print('Not supported platform!')

  if save_flag == 0:
    os.remove(filename)
  else:
    pass


def sound(x, fs, filename=None):
  '''
    sound(x, fs, filename=None)
      Function for playing audio signal to listen. Only supports mac and linux.

    arguments
      x: Audio signal.
      fs: Sample rate of the audio signal x.
      filename (optional): Name of the audio file to save. Has to end with '.wav'.
                           This is required only if you want to remain the audio file.

    return
      Nothing is returned but the sound is played.
  '''

  # The audio file remains only if 'filename' is given.
  save_flag = 1
  if filename is None:
    save_flag = 0
    filename = 'cast_tmp.wav'

  assert filename.endswith('.wav'), 'Only wav format is supported!'

  write(filename, fs, x)
  # now that file is written to the disk - play it
  if platform.system() == 'Linux':
    subprocess.call(["play", filename]) #<-  for linux
  elif platform.system() == 'Darwin':
    subprocess.call(["afplay", filename]) #<- for Mac
  else:
    print('Not supported platform!')

  if save_flag == 0:
    os.remove(filename)
  else:
    pass


def play(filename=None):
  '''
    play(filename=None)
      Function for playing audio file to listen. Only supports mac and linux.

    arguments
      filename: Name of the audio file to save. Has to end with '.wav'.
                This is required only if you want to remain the audio file.

    return
      Nothing is returned but the sound is played.
  '''

  if filename is None:
    return 0

  assert filename.endswith('.wav'), 'Only wav format is supported!'

  if platform.system() == 'Linux':
    subprocess.call(["play", filename]) #<-  for linux
  elif platform.system() == 'Darwin':
    subprocess.call(["afplay", filename]) #<- for Mac
  else:
    print('Not supported platform!')


def audio_to_npy(audio_folder='./bg', target_fs=22050, delete_audio=1):
  '''
    audio_to_npy(audio_folder='bg', target_fs=22050, delete_audio=1)
      Function for transforming audio files ('.wav' or '.mp3') into '.npy' format.
      This function mainly targets transforming background sounds.

    arguments
      audio_folder: Folder that contains audio files to be transformed.
      target_fs: Target sample rate. Default = 22050 Hz.
                 If the original sample rate is different from 'target_fs', 
                 the audio is resampled before it is saved.
      delete_audio: Delete the audio files in the 'audio_folder' if this is 1. (Default)
                    Else, the audio files remain.

    return
      Nothing is returned. '.npy' files remain saved in the 'audio_folder'.
  '''

  if sys.version_info[0] == 2:
    (loc, _, filenames) = os.walk(audio_folder).next()
  elif sys.version_info[0] == 3:
    (loc, _, filenames) = os.walk(audio_folder).__next__()
  else:
    print('Unsupported python version!')
    (loc, _, filenames) = os.walk(audio_folder).__next__()

  filenames = [f for f in filenames if not f.startswith('.')]
  
  for filename in filenames:
    if filename.endswith('.wav') or filename.endswith('.mp3'):
      # load audio
      x_bg, fs_bg = librosa.core.load(loc+'/'+filename)
      if fs_bg != target_fs:
        x_bg = librosa.resample(x_bg, fs_bg, target_fs)
        fs_bg = target_fs

      # name of the file to save
      npyname = filename[:-4]+'.npy'
      np.save(os.path.join(loc, npyname), x_bg)

      if delete_audio==1:
        # delete the audio file
        os.remove(loc+'/'+filename)


def cut_event(x, fs, target_duration=None):
  '''
    cut_event(x, fs, target_duration=None)
      Function that cuts the input audio (=event, expectedly)
      to have 'target_duration'.

    arguments
      x: Input audio (possibly sound event).
      fs: Sample rate.
      target_duration: Expected duration of the output in seconds.

    return
      x_cut: x that was randomly cut to have target_duration. 
  '''
  
  if target_duration is None:
    print('Error! target_duration not provided!')
    return 0
  else:
    duration_sample = int(target_duration*fs)

  if duration_sample < len(x):  # Most case
    total_margin = len(x)-duration_sample
    front_margin = np.random.randint(total_margin)
    x_cut = x[front_margin:front_margin+duration_sample]
    return x_cut
  
  elif duration_sample == len(x):
    x_cut = x
    return x_cut

  else:
    total_margin = duration_sample-len(x)
    front_margin = np.random.randint(total_margin)
    x_cut = np.zeros((duration_sample))
    x_cut[front_margin:front_margin+len(x)] += x
    return x_cut


