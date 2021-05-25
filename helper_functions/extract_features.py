import os
import numpy as np

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

path_to_files = 'path/to/wav/files'
destination = 'features/path'

for filename in os.listdir(path_to_files):
    if '.wav' in filename:
        (rate, sig) = wav.read(os.path.join(path_to_files, filename))
        fbank_feat = logfbank(sig, rate, nfilt=60)

        filename = filename.split('.wav')[0]

        np.save(os.path.join(destination, filename), fbank_feat)
