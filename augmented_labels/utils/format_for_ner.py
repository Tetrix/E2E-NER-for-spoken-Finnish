import os


for filename in os.listdir('/m/triton/scratch/elec/puhe/p/porjazd1/E2E_NER/E2E-NER/augmented_labels/data/transcripts/temp'):
    if os.path.isfile(os.path.join('/m/triton/scratch/elec/puhe/p/porjazd1/E2E_NER/E2E-NER/augmented_labels/data/transcripts/temp', filename)):
        with open(os.path.join('/m/triton/scratch/elec/puhe/p/porjazd1/E2E_NER/E2E-NER/augmented_labels/data/transcripts/temp', filename), 'r') as f:
            data = f.readlines()

            with open(os.path.join('/m/triton/scratch/elec/puhe/p/porjazd1/E2E_NER/E2E-NER/augmented_labels/data/transcripts/formatted', filename), 'w') as o:
                for sent in data:
                    sent = sent.split()
                    for word in sent:
                        o.write(word + '\n')
                    o.write('\n\n')
