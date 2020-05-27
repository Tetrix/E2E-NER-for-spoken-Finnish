import os


for filename in os.listdir('../data/transcripts/train_small'):
    if os.path.isfile(os.path.join('../data/transcripts/train_small', filename)):
        with open(os.path.join('../data/transcripts/train_small/', filename), 'r') as f:
            data = f.readlines()

            with open(os.path.join('../data/transcripts/ner_transcripts_train/temp', filename), 'w') as o:
                for sent in data:
                    sent = sent.split()
                    for word in sent:
                        o.write(word + '\n')
                    o.write('\n\n')
