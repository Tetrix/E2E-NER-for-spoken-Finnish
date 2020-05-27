import os

dir_path = os.path.dirname(os.path.realpath(__file__))
source = os.path.join(dir_path, '../../data/transcripts/dev_small')

for filename in sorted(os.listdir(source)):
    with open(os.path.join('../data/transcripts/dev_ner/temp', filename), 'w') as f:
        with open(os.path.join(source, filename), 'r') as r:
            data = r.readlines()
            data = data[0].split()
            
            for word in data:
                f.write(word + '\n')
            f.write('\n')

            
