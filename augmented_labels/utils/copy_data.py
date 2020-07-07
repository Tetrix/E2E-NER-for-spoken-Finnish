import os
from shutil import copy
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np


current_dir = os.path.dirname(os.path.realpath(__file__))
destination_transcripts = os.path.join(current_dir, '/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/transcripts/whole')
destination_audio = os.path.join(current_dir, '../data/audio/whole')


with open('../data/text_whole', 'r') as f:
    data = f.readlines()

with open('../data/wav_whole.scp') as f:
    audio = f.readlines()


# copy audio and transcripts
#for i in range(len(audio)):
#    line = audio[i].rstrip()
#    audio_name = line.split()[0]
#    for j in range(len(data)):
#        if audio_name+'-1 ' in data[j]:
#            name = data[j].split()[0]
#            transcript = data[j].split()[1:]
#            transcript = ' '.join(transcript)

#            with open(os.path.join(destination_transcripts, name + '.txt'), 'w') as f:
#                f.write(transcript)

            #copy audio
            #sample = line.split()[1][1:]
            #source = os.path.join('/m/triton', sample)
            #copy(source, destination_audio)





# extract features and transcripts without copying audio
for i in range(len(audio)):
    line = audio[i].rstrip()
    audio_name = line.split()[0]
    for j in range(len(data)):
        if audio_name+'-1 ' in data[j]:
            name = data[j].split()[0]
            transcript = data[j].split()[1:]
            transcript = ' '.join(transcript)

            with open(os.path.join(destination_transcripts, name + '.txt'), 'w') as f:
               f.write(transcript)

            #copy audio
            sample = line.split()[1][1:]
            source = os.path.join('/m/triton', sample)

            (rate, sig) = wav.read(source)
            fbank_feat = logfbank(sig, rate, nfilt=60)

            np.save(os.path.join('/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/features/whole', name), fbank_feat)





