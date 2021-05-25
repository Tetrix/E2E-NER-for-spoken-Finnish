import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle
import fasttext
import time

import utils.prepare_data as prepare_data
from model import Encoder, Decoder, DecoderNER
from config.config import *
from train import train
from get_predictions import get_predictions


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Libri clean')
print(device)

# load features and labels
print('Loading data..')


# Swedish data
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/train.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/swedish/train.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/swedish/ner_train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/swedish/dev.txt')
#tags_dev = prepare_data.load_tags('../augmented_labels/data/normalized/ner/swedish/ner_dev.txt')



# LibriSpeech data
features_train = prepare_data.load_features('../augmented_labels/data/normalized/features/libri/train')
target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/libri/train.txt')
tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/libri/ner_train.txt')

features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/libri/dev.npy')
target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/libri/dev.txt')
tags_dev = prepare_data.load_tags('../augmented_labels/data/normalized/ner/libri/ner_dev.txt')


# English-Gold data
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/train.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/eng_ood/train.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/eng_ood/ner_train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/eng_ood/dev.txt')
#tags_dev = prepare_data.load_tags('../augmented_labels/data/normalized/ner/eng_ood/ner_dev.txt')



print('Done...')

print('Loading embeddings...')
#embeddings = fasttext.load_model('weights/embeddings/cc.sv.300.bin')
embeddings = fasttext.load_model('weights/embeddings/crawl-300d-2M-subword.bin')
print('Done...')


# generate index dictionaries
#char2idx, idx2char = prepare_data.encode_data(target_train)

# generate index dictionaries
#with open('weights/char2idx_swe.pkl', 'wb') as f:
#    pickle.dump(char2idx, f, protocol=pickle.HIGHEST_PROTOCOL)

#with open('weights/idx2char_swe.pkl', 'wb') as f:
#    pickle.dump(idx2char, f, protocol=pickle.HIGHEST_PROTOCOL)


# used for normalized
tag2idx = {'O': 1, 'PER': 2, 'LOC': 3, 'ORG': 4}

idx2tag = {1: 'O', 2: 'PER', 3: 'LOC', 4: 'ORG'}


with open('weights/char2idx_libri.pkl', 'rb') as f:
    char2idx = pickle.load(f)
with open('weights/idx2char_libri.pkl', 'rb') as f:
    idx2char = pickle.load(f)


char2idx['~'] = len(char2idx) + 1
idx2char[len(idx2char) + 1] = '~'


char2idx_ctc = {}
idx2char_ctc = {}
counter = 0
for key, value in char2idx.items():
    if value >= 4:
        char2idx_ctc[key] = counter
        idx2char_ctc[counter] = key
        counter += 1


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



# initialize the Encoder
encoder = Encoder(features_train[0].size(1), encoder_hidden_size, encoder_layers, len(char2idx_ctc), batch_size, device).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)

# initialize the Decoder
decoder = Decoder(embedding_dim_chars, encoder_hidden_size, attention_hidden_size, num_filters, len(char2idx)+1, decoder_layers, encoder_layers, batch_size, attention_type, device).to(device)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)

# initialize the DecoderNER
decoder_ner = DecoderNER(embedding_dim_words, decoder_ner_hidden_size, len(tag2idx)+1, device).to(device)
decoder_ner_optimizer = optim.Adam(decoder_ner.parameters(), lr=decoder_ner_lr)


print(encoder)
print(decoder)

total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_trainable_params_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
total_trainable_params_decoder_ner = sum(p.numel() for p in decoder_ner.parameters() if p.requires_grad)


print('The number of trainable parameters is: %d' % (total_trainable_params_encoder + total_trainable_params_decoder + total_trainable_params_decoder_ner))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load('weights/english_finetuned/state_dict_4.pt')
    #encoder.load_state_dict(checkpoint['encoder'])
    #decoder.load_state_dict(checkpoint['decoder'])
    #decoder_ner.load_state_dict(checkpoint['decoder_ner'])
    #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    #decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
    #decoder_ner_optimizer.load_state_dict(checkpoint['decoder_ner_optimizer'])

    # freeze the weights of the encoder and the ASR decoder
    #for param in encoder.parameters():
    #    param.requires_grad = False

    #for param in decoder.parameters():
    #    param.requires_grad = False



    criterion = nn.NLLLoss(ignore_index=0, reduction='mean')
    ctc_loss = torch.nn.CTCLoss(blank=char2idx_ctc['~'], reduction='mean', zero_infinity=True)
    train(pairs_batch_train, 
            pairs_batch_dev, 
            encoder, 
            decoder,
            decoder_ner, 
            encoder_optimizer, 
            decoder_optimizer, 
            decoder_ner_optimizer, 
            criterion, 
            ctc_loss, 
            batch_size, 
            num_epochs, 
            device, 
            len(train_data), 
            len(dev_data))
else:
    checkpoint = torch.load('weights/libri_big/state_dict_18.pt', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    decoder_ner.load_state_dict(checkpoint['decoder_ner'])
