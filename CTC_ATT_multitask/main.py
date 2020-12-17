import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle
import gensim
import fasttext

import utils.prepare_data as prepare_data
from utils.radam import RAdam
from model import Encoder, Decoder, DecoderNER
from config.config import *
from train import train
from get_predictions import get_predictions


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load language model related stuff
#from utils.language_model_eng.model import CharRNN
#from utils.language_model_eng.helpers import *
#
#language_model = CharRNN(n_characters, 1024, n_characters, 'lstm', 2)
#language_model.load_state_dict(torch.load('utils/language_model_eng/lm_weights/language_model_eng.pt', map_location='cpu'))
#language_model = language_model.to(device)
#language_model.eval()

from utils.language_model_eng.new_lm.model import CharRNN
with open('utils/language_model_eng/new_lm/data/whole.txt', 'r') as f:
    text = f.read()
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
# Define and print the net

language_model = CharRNN(chars, 1024, 3).to(device)
language_model.load_state_dict(torch.load('utils/language_model_eng/new_lm/models/1024_3_20.pt', map_location='cpu'))
language_model.eval()



print('libri')
print(device)

# load features and labels
print('Loading data..')


# Parliament data
#features_train = prepare_data.load_features('../augmented_labels/data/normalized/features/train')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/train.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/ner_train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/dev.txt')
#tags_dev = prepare_data.load_tags('../augmented_labels/data/normalized/ner/ner_dev.txt')



# Swedish data
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/train.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/swedish/train.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/swedish/ner_train.txt')
#
#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/swedish/dev.txt')
#tags_dev = prepare_data.load_tags('../augmented_labels/data/normalized/ner/swedish/ner_dev.txt')



# LibriSpeech data
#features_train = prepare_data.load_features('../augmented_labels/data/normalized/features/libri/train')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/libri/train.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/libri/ner_train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/libri/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/libri/dev.txt')
#tags_dev = prepare_data.load_tags('../augmented_labels/data/normalized/ner/libri/ner_dev.txt')




# test Parliament
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/test.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/test.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/ner_test.txt')



# test Swedish
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/test.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/swedish/test.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/swedish/ner_test.txt')


# test LibriSpeech
features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/libri/test_clean.npy')
target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/libri/test_clean.txt')
tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/libri/ner_test_clean.txt')



# test English out of domain
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_test/test_eng.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/eng_test/test_eng.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/eng_test/ner_test_eng.txt')


features_train = features_train[:100]
target_train = target_train[:100]
tags_train = tags_train[:100]

features_dev = features_train
target_dev = target_train
tags_dev = tags_train


print('Done...')

print('Loading embeddings...')
#embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
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

print('The number of trainable parameters is: %d' % (total_trainable_params_encoder + total_trainable_params_decoder))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load('weights/swedish_small/state_dict_1.pt')
    #encoder.load_state_dict(checkpoint['encoder'])
    #decoder.load_state_dict(checkpoint['decoder'])
    #decoder_ner.load_state_dict(checkpoint['decoder_ner'])
    #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    #decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
    #decoder_ner_optimizer.load_state_dict(checkpoint['decoder_ner_optimizer'])

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
    checkpoint = torch.load('weights/libri/state_dict_18.pt', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    decoder_ner.load_state_dict(checkpoint['decoder_ner'])



batch_size = 1

pairs_batch_train = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions(encoder, decoder, language_model, decoder_ner, batch_size, idx2char, idx2tag, pairs_batch_train, MAX_LENGTH)

