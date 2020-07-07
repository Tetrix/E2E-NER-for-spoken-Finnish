import os

dir_path = os.path.dirname(os.path.realpath(__file__))
source = os.path.join(dir_path, '/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/transcripts/train')

for filename in sorted(os.listdir(source)):
    with open(os.path.join('/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/ner/ner_formatted_train', filename), 'w') as f:
        with open(os.path.join(source, filename), 'r') as r:
            data = r.readlines()
            data = data[0].split()
            
            for word in data:
                f.write(word + '\n')
            f.write('\n')

            
