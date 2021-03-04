# test set generator
# using CAST library
# jspark@cochlear.ai

import librosa
import numpy as np
import cast
import os


testdata_folder = '../test_files'
duration = 6  # in sec
music_folder = '../songs'
noise_folder = '../pub_noise'
ir_folder = '../ir'
n_test = 500  # 0 denotes the number of all songs in music_folder
target_fs = 22050
# snr_range=(10, 10)  # in dB
snr_list = [0, 3, 6, 10]




def my_cut_event(x, fs, target_duration=None):
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
    return (x_cut, front_margin)
  
  elif duration_sample == len(x):
    x_cut = x
    return (x_cut, 0)

  else:
    total_margin = duration_sample-len(x)
    front_margin = np.random.randint(total_margin)
    x_cut = np.zeros((duration_sample))
    x_cut[front_margin:front_margin+len(x)] += x
    return (x_cut, 0)






try:
  os.mkdir(testdata_folder)
except:
  pass

(path, _, filenames) = os.walk(music_folder).__next__()
filenames = [x for x in filenames if x.endswith('.mp3') or x.endswith('.wav')]

if n_test == 0:
  n_test = len(filenames)

cnt = 0
for filename in filenames: # music file
  print(filename)
  if cnt >= n_test:
    break
  else:
    cnt += 1
  
  # read the music and cut
  x, fs = librosa.core.load(os.path.join(path,filename), sr=target_fs, mono=True, res_type='kaiser_fast')
  x_cut, front_margin_samples = my_cut_event(x, fs, target_duration=duration)  # not cast.cut_event
  front_margin_sec = round(front_margin_samples/fs,1)
  E_x = cast.energy_measure(x_cut)/float(len(x_cut))  # energy normalize
  x_cut = x_cut/np.sqrt(E_x)

  # draw a background noise
  (_, _, noise_names) = os.walk(noise_folder).__next__()
  noise_names.sort()
  noise_idx = np.random.randint(len(noise_names))
  x_bg = np.load(os.path.join(noise_folder,noise_names[noise_idx]))
  x_bg = cast.cut_event(x_bg, target_fs, target_duration=duration)
  E_bg = cast.energy_measure(x_bg)/float(len(x_bg))  # energy normalize
  x_bg = x_bg/np.sqrt(E_bg)

  # draw a impulse response
  (_, _, ir_names) = os.walk(ir_folder).__next__()
  ir_names.sort()
  ir_idx = np.random.randint(len(ir_names))
  x_ir = np.load(os.path.join(ir_folder,ir_names[ir_idx]))
  
  # mix & ir aug & save
  for snr in snr_list:
    magnitude = np.power(10,snr/20.0) # this includes sqrt
    x_mix = x_bg+magnitude*x_cut
    x_mix = cast.max_normalise(x_mix)
    if len(x_mix) < len(x_ir):
      x_ir = x_ir[:len(x_mix)]
    elif len(x_mix) > len(x_ir):
      x_ir = np.concatenate((x_ir, np.zeros(len(x_mix)-len(x_ir))),axis=0)
    fftlen = np.maximum(len(x_mix),len(x_ir))
    X = np.fft.fft(x_mix,n=fftlen)
    X_ir = np.fft.fft(x_ir,n=fftlen)
    x_aug = np.fft.ifft(np.multiply(X_ir,X))[:len(x_mix)].real
    x_aug = x_aug/np.max(np.abs(x_aug))
    
    # save to file
    try:
      os.mkdir(os.path.join(testdata_folder,str(snr)))
    except:
      pass
    #librosa.output.write_wav(os.path.join(testdata_folder,filename[:-4]+'_'+str(front_margin_sec)+'sec_'+str(snr)+'dB.wav'), x_aug, fs)
    librosa.output.write_wav(os.path.join(testdata_folder,str(snr),'{}_{}sec_{}noise_{}ir_{}dB.wav'.format(filename[:-4],front_margin_sec,noise_idx,ir_idx,snr)), x_aug, fs)


