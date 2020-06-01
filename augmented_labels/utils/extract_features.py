import os
import numpy as np

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

current_dir = os.path.dirname(os.path.realpath(__file__))

for filename in os.listdir(os.path.join(current_dir, '../data/audio/whole_small')):
    if '.wav' in filename:
        (rate, sig) = wav.read(os.path.join('../data/audio/whole_small', filename))
        fbank_feat = logfbank(sig, rate, nfilt=60)

        filename = filename.split('.wav')[0]

        np.save(os.path.join(current_dir, '../data/features/whole_small', filename), fbank_feat)
