import os
from shutil import copy

current_dir = os.path.dirname(os.path.realpath(__file__))
destination_transcripts = os.path.join(current_dir, '../data/transcripts/whole_small')
destination_audio = os.path.join(current_dir, '../data/audio/whole_small')


with open('../data/text.txt', 'r') as f:
    data = f.readlines()

with open('../data/wav.scp') as f:
    audio = f.readlines()



for i in range(len(audio)):
    if i < 222000:
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
                copy(source, destination_audio)
    else:
        break
