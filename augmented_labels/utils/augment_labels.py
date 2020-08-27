import sys, os

#dir_path = '/m/triton/scratch/elec/puhe/p/porjazd1/E2E_NER/E2E-NER/augmented_labels'

for filename in os.listdir('/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/transcripts/normalized/train'):
    with open(os.path.join('/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/augmented/normalized/train', filename), 'w') as f:
        
        with open(os.path.join('/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/transcripts/normalized/train', filename), 'r') as ff:
            labels = ff.readlines()
        
        with open(os.path.join('/m/teamwork/t40511_asr/p/NER/datasets/parliament_whole/ner/normalized/train', filename), 'r') as ff:
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



