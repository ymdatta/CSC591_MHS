#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import noisereduce as nr
import csv
import soundfile as sf
import pickle


is_Mohan = True

if is_Mohan:
  audio_dir = '/content/drive/MyDrive/MHA_Proj_Data/Project_Dataset_Release/AUDIO'
  metadata_path = '/content/drive/MyDrive/MHA_Proj_Data/Project_Dataset_Release/metadata.csv'
  test_base_path = '/content/drive/MyDrive/MHA_Proj_Data/Project_Dataset_Release/LISTS/'

  audio_dir = '/home/myelugo/mhs/Project_Dataset_Release/AUDIO'
  metadata_path = '/home/myelugo/mhs/Project_Dataset_Release/metadata.csv'
  test_base_path = '/home/myelugo/mhs/Project_Dataset_Release/LISTS/'
else:
  audio_dir = '/content/drive/MyDrive/Project_Dataset_Release/AUDIO'
  metadata_path = '/content/drive/MyDrive/Project_Dataset_Release/metadata.csv'

id_to_gender = {}
id_to_status = {}

with open(metadata_path, newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        row_split = row[0].split(' ')
        if row_split[0] == 'SUB_ID':
            continue
        id_to_gender[row_split[0]] = row_split[2]
        id_to_status[row_split[0]] = row_split[1]


# In[ ]:

id_to_gender['FONIOYTJ']


# In[ ]:


train_files = ['train_0.csv', 'train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']
valid_files = ['val_0.csv', 'val_1.csv', 'val_2.csv', 'val_3.csv', 'val_4.csv']

with open('temp.pickle', 'wb') as handle:
   pickle.dump(id_to_gender, handle, protocol=pickle.HIGHEST_PROTOCOL)

def file_to_IDs(file):
    file = test_base_path + file
    res = []

    with open(file, newline='\n') as file:
      reader = csv.reader(file)
      for row in reader:
        res.append(row[0])

    return res


# In[ ]:


"""
Step 1: You are given an ID:
Step 2: Get three audio samples for the ID from AUDIO's breathing, cough, and speech.
Step 3: Extract features from each samples.
Step 3.1: If it's a positive sample, do augmentation and extract features. Why? To balance +ve and -ve samples.
Step 4: Return a list of elements. Each element is further a list with two elements (feature vector and status value)
"""
Datadir = audio_dir


# In[ ]:


"""
Data Augmentation ideas:

1. Time strech. (Twice as fast and half the original speed)
2. Pitch shift. (Shift by a tritone for now)
"""

def augmentSignal(y, sr):
    try:
      y_fast = librosa.effects.time_stretch(y, rate=2.0)
      y_slow = librosa.effects.time_stretch(y, rate=0.5)
      y_tritone = librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)
    except:
      return None
    return [y, y_fast, y_slow, y_tritone]

def intervalsToSignals(intervals, y):
    n, _ = intervals.shape
    res = []
    for i in range(0, n):
        res.append(y[intervals[i][0]:intervals[i][1]])
    return res

def displaySignal(y):
    plt.figure()
    librosa.display.waveshow(y=y)
    plt.xlabel("Time (xsecs) ")
    plt.ylabel("Amplitude")
    plt.show()

def ID_to_featureVector(ID, train=False):
    breathing_y, breathing_s = librosa.load(Datadir + '/breathing/' + ID + '.flac')
    cough_y, cough_s = librosa.load(Datadir + '/cough/' + ID + '.flac')
    speech_y, speech_s = librosa.load(Datadir + '/speech/' + ID + '.flac')

    # Denoise the signal
    breathing_y = nr.reduce_noise(y=breathing_y, sr=breathing_s)
    cough_y = nr.reduce_noise(y=cough_y, sr=cough_s)
    speech_y = nr.reduce_noise(y=speech_y, sr=speech_s)

    res = []
    is_f = 1 if (id_to_gender[ID] == 'm') else 0

    ## Augment only if we are training, not during testing!
    if train:
      # If it belongs to a positive patient, augment the signal and split it into non-silent intervals.
      # [[non-silent signal arrays], sr, male or female information, covid positive or negative information]
      if (id_to_status[ID] == 'p'):
          for (y, sr) in [(breathing_y, breathing_s), (cough_y, cough_s), (speech_y, speech_s)]:
              aug_y = augmentSignal(y, sr)
              if aug_y == None:
                continue
              for y_sig in aug_y:
                  intervals = librosa.effects.split(y_sig)
                  sSplit = intervalsToSignals(intervals, y_sig)
                  for signal in sSplit:
                      res.append([signal, sr, is_f])
      else:
          for (y, sr) in [(breathing_y, breathing_s), (cough_y, cough_s), (speech_y, speech_s)]:
              intervals = librosa.effects.split(y)
              sSplit = intervalsToSignals(intervals, y)
              for signal in sSplit:
                  res.append([signal, sr, is_f])
    else:
        for (y, sr) in [(breathing_y, breathing_s), (cough_y, cough_s), (speech_y, speech_s)]:
          aug_y = augmentSignal(y, sr)
          if aug_y == None:
            continue
          for y_sig in aug_y:
            intervals = librosa.effects.split(y_sig)
            sSplit = intervalsToSignals(intervals, y_sig)
            for signal in sSplit:
                res.append([signal, sr, is_f])

    return res

"""
Obs1: Takes a lot of time to load an ID's three sounds.
Obs2: Have everthing at hand, so they maybe used again in the future.
"""

"""
Extract Features from Processed Signals
"""
def extract_features(signal, sr):
    features = {}
    # Time-domain features
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(signal))
    features['rms'] = np.mean(librosa.feature.rms(y=signal))

    # Frequency-domain features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))

    # Mel-frequency cepstral coefficients (MFCC)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i+1}'] = np.mean(mfcc)

    return features

"""
Apply Feature Extraction to All Processed Signals
Use the output of ID_to_featureVector to extract features for every audio segment.
"""
def create_feature_dataset(processed_signals):
    dataset = []
    for signal, sr, is_f in processed_signals:
        # Extract features for the current signal
        features = extract_features(signal, sr)
        features['is_female'] = is_f
        dataset.append(features)

    return pd.DataFrame(dataset)

train_features = {}
valid_features = {}

print("Starting building features!")

for ID in list(id_to_gender.keys()):
  train_features[ID] = create_feature_dataset(ID_to_featureVector(ID, train=True))
  valid_features[ID] = create_feature_dataset(ID_to_featureVector(ID, train=False))


with open('train_v1.pickle', 'wb') as handle:
   pickle.dump(train_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('valid_v1.pickle', 'wb') as handle:
   pickle.dump(valid_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Completed!")
