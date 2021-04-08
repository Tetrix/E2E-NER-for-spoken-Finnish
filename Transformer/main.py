import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle
import gensim
import fasttext
import math

import utils.prepare_data as prepare_data
from utils.radam import RAdam
from model import Encoder, Decoder, Transformer
from config.config import *
from train import train
from get_predictions import get_predictions


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.autograd.set_detect_anomaly(True)


print(device)

# load features and labels
print('Loading data..')


# Parliament data
#features_train = prepare_data.load_features('../NoDaLiDa/augmented_labels/data/normalized/features/train', max_len)
#target_train = prepare_data.load_transcripts('../NoDaLiDa/augmented_labels/data/normalized/transcripts/train.txt')
#tags_train = prepare_data.load_tags('../NoDaLiDa/augmented_labels/data/normalized/ner/ner_train.txt')
#
#features_dev = prepare_data.load_features_combined('../NoDaLiDa/augmented_labels/data/normalized/features/dev.npy', max_len)
#target_dev = prepare_data.load_transcripts('../NoDaLiDa/augmented_labels/data/normalized/transcripts/dev.txt')
#tags_dev = prepare_data.load_tags('../NoDaLiDa/augmented_labels/data/normalized/ner/ner_dev.txt')



# Swedish data
#features_train = prepare_data.load_features_combined('../NoDaLiDa/augmented_labels/data/normalized/features/swedish/train.npy', max_len)
#target_train = prepare_data.load_transcripts('../NoDaLiDa/augmented_labels/data/normalized/transcripts/swedish/train.txt')
#tags_train = prepare_data.load_tags('../NoDaLiDa/augmented_labels/data/normalized/ner/swedish/ner_train.txt')
#
#features_dev = prepare_data.load_features_combined('../NoDaLiDa/augmented_labels/data/normalized/features/swedish/dev.npy', max_len)
#target_dev = prepare_data.load_transcripts('../NoDaLiDa/augmented_labels/data/normalized/transcripts/swedish/dev.txt')
#tags_dev = prepare_data.load_tags('../NoDaLiDa/augmented_labels/data/normalized/ner/swedish/ner_dev.txt')



# LibriSpeech data
#features_train = prepare_data.load_features('../NoDaLiDa/augmented_labels/data/normalized/features/libri/train', max_len)
#target_train = prepare_data.load_transcripts('../NoDaLiDa/augmented_labels/data/normalized/transcripts/libri/train.txt')
#tags_train = prepare_data.load_tags('../NoDaLiDa/augmented_labels/data/normalized/ner/libri/ner_train.txt')
#
#features_dev = prepare_data.load_features_combined('../NoDaLiDa/augmented_labels/data/normalized/features/libri/dev.npy', max_len)
#target_dev = prepare_data.load_transcripts('../NoDaLiDa/augmented_labels/data/normalized/transcripts/libri/dev.txt')
#tags_dev = prepare_data.load_tags('../NoDaLiDa/augmented_labels/data/normalized/ner/libri/ner_dev.txt')




# test Parliament
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/test.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/test.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/ner_test.txt')



# test Swedish
features_train = prepare_data.load_features_combined('../NoDaLiDa/augmented_labels/data/normalized/features/swedish/test.npy', max_len)
target_train = prepare_data.load_transcripts('../NoDaLiDa/augmented_labels/data/normalized/transcripts/swedish/test.txt')
tags_train = prepare_data.load_tags('../NoDaLiDa/augmented_labels/data/normalized/ner/swedish/ner_test.txt')


# test LibriSpeech
#features_train = prepare_data.load_features_combined('../NoDaLiDa/augmented_labels/data/normalized/features/libri/test_clean.npy', max_len)
#target_train = prepare_data.load_transcripts('../NoDaLiDa/augmented_labels/data/normalized/transcripts/libri/test_clean.txt')
#tags_train = prepare_data.load_tags('../NoDaLiDa/augmented_labels/data/normalized/ner/libri/ner_test_clean.txt')



# test English out of domain
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_test/test_eng.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/eng_test/test_eng.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/eng_test/ner_test_eng.txt')


features_train = features_train[:200]
target_train = target_train[:200]
tags_train = tags_train[:200]


features_dev = features_train
target_dev = target_train
tags_dev = tags_train


print('Done...')

print('Loading embeddings...')
#embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
embeddings = fasttext.load_model('weights/embeddings/cc.sv.300.bin')
#embeddings = fasttext.load_model('weights/embeddings/crawl-300d-2M-subword.bin')
print('Done...')


# generate index dictionaries
#char2idx, idx2char = prepare_data.encode_data(target_train)

# generate index dictionaries
#with open('weights/char2idx_libri.pkl', 'wb') as f:
#    pickle.dump(char2idx, f, protocol=pickle.HIGHEST_PROTOCOL)

#with open('weights/idx2char_libri.pkl', 'wb') as f:
#    pickle.dump(idx2char, f, protocol=pickle.HIGHEST_PROTOCOL)


# used for normalized
tag2idx = {'O': 1, 'PER': 2, 'LOC': 3, 'ORG': 4}
idx2tag = {1: 'O', 2: 'PER', 3: 'LOC', 4: 'ORG'}


with open('weights/char2idx_swe.pkl', 'rb') as f:
    char2idx = pickle.load(f)
with open('weights/idx2char_swe.pkl', 'rb') as f:
    idx2char = pickle.load(f)


#char2idx = {}
#char2idx['<sos>'] = 1
#char2idx['<eos>'] = 2
#char2idx['UNK'] = 3
#char2idx[' '] = 4
#
#import string
#chars = string.ascii_lowercase + 'å' + 'ä' + 'ö'
#
#for i in chars:
#    if i not in char2idx.keys():
#        char2idx[i] = len(char2idx) + 1
#
#idx2char = {v: k for k, v in char2idx.items()}


# convert labels to indices
indexed_target_train = prepare_data.label_to_idx(target_train, char2idx)
indexed_target_dev = prepare_data.label_to_idx(target_dev, char2idx)

indexed_target_word_train = prepare_data.word_to_idx(target_train, embeddings)
indexed_target_word_dev = prepare_data.word_to_idx(target_dev, embeddings)

indexed_tags_train = prepare_data.tag_to_idx(tags_train, tag2idx)
indexed_tags_dev = prepare_data.tag_to_idx(tags_dev, tag2idx)


# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train, indexed_target_train, indexed_target_word_train, indexed_tags_train)
dev_data = prepare_data.combine_data(features_dev, indexed_target_dev, indexed_target_word_dev, indexed_tags_dev)


# remove extra data that doesn't fit in batch
train_data = prepare_data.remove_extra(train_data, batch_size)
dev_data = prepare_data.remove_extra(dev_data, batch_size)


pairs_batch_train = DataLoader(dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)

pairs_batch_dev = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)



transformer = Transformer(features_train[0].size(1), len(char2idx)+1, n_head_encoder, n_head_decoder, d_model, n_layers_encoder, n_layers_decoder, max_len).to(device)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


total_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print('The number of trainable parameters is: %d' % (total_trainable_params))


# train
if skip_training == False:
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr) 

    #checkpoint = torch.load('weights/libri_big/state_dict_4.pt', map_location=torch.device('cpu'))
    #transformer.load_state_dict(checkpoint['transformer'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    train(pairs_batch_train, 
            pairs_batch_dev,
            transformer,
            criterion,
            optimizer,
            num_epochs,
            batch_size,
            len(features_train),
            len(features_dev),
            device) 
else:
    checkpoint = torch.load('weights/swedish/state_dict_20.pt', map_location=torch.device('cpu'))
    transformer.load_state_dict(checkpoint['transformer'])



batch_size = 1

pairs_batch_train = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions(transformer, batch_size, idx2char, idx2tag, pairs_batch_train)
