import os

source = '/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/transcripts/normalized/test'

for filename in os.listdir(source):
    if os.path.isfile(os.path.join(source, filename)):
        with open(os.path.join(source, filename), 'r') as f:
            data = f.readlines()

            with open(os.path.join('/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/ner/normalized/ner_formatted_test', filename), 'w') as o:
                for sent in data:
                    sent = sent.split()
                    for word in sent:
                        o.write(word + '\n')
                    o.write('\n\n')
