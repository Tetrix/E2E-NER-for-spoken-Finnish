import os
import numpy as np

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

current_dir = os.path.dirname(os.path.realpath(__file__))

for filename in os.listdir('audio/'):
    if '.wav' in filename:
        (rate, sig) = wav.read(os.path.join('audio', filename))
        fbank_feat = logfbank(sig, rate, nfilt=40)
        fbank_feat -= (np.mean(fbank_feat, axis=0) + 1e-8)

        filename = filename.split('.wav')[0]

        np.save(os.path.join('features/whole_40', filename), fbank_feat)
