import sys, os

dir_path = '/m/triton/scratch/elec/puhe/p/porjazd1/E2E_NER/E2E-NER/augmented_labels'

for filename in os.listdir('../data/transcripts/train_small'):
    with open(os.path.join(dir_path, 'data/transcripts/augmented/augmented_train_small', filename), 'w') as f:
        
        with open(os.path.join(dir_path, 'data/transcripts/train_small', filename), 'r') as ff:
            labels = ff.readlines()
        
        with open(os.path.join(dir_path, 'data/transcripts/ner_transcripts_train', filename), 'r') as ff:
            tags = ff.readlines()
        
        result = []
        labels = labels[0].split()
        for i in range(len(labels)):
            if tags[i] != '\n':
                label = labels[i]
                tag = tags[i].rstrip()

            result.append(label)
            result.append(tag)

        result = ' '.join(result)
        
        f.write(result)



