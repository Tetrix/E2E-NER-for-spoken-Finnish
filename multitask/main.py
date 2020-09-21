import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

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

from utils.language_model.model import CharRNN
from utils.language_model.helpers import *


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# load language model related stuff
language_model = CharRNN(n_characters, 1024, n_characters, 'lstm', 2)
language_model.load_state_dict(torch.load('utils/language_model/lm_weights/language_model_1024_10000.pt', map_location='cpu'))
language_model = language_model.to(device)
language_model.eval()


# load features and labels
print('Loading data..')

# whole data normalized
#features_train = prepare_data.load_features('../augmented_labels/data/normalized/features/train_small')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/train_small.txt')
#tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/ner_train_small.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/dev_small.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/dev_small.txt')
#tags_dev = prepare_data.load_tags('../augmented_labels/data/normalized/ner/ner_dev_small.txt')


# test normalized
features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/test.npy')
target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/test.txt')
tags_train = prepare_data.load_tags('../augmented_labels/data/normalized/ner/ner_test.txt')


# compare againt conventional NER
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/test.npy')
#target_train = prepare_data.load_transcripts('output/e2e_asr_combined.txt')
#tags_train = prepare_data.load_tags('output/conventional_ner.txt')


features_train = features_train[:5000]
target_train = target_train[:5000]
tags_train = tags_train[:5000]

features_dev = features_train
target_dev = target_train
tags_dev = tags_train

print('Done...')

print('Loading embeddings...')
#embeddings = gensim.models.KeyedVectors.load_word2vec_format('weights/embeddings/fin-word2vec.bin', binary=True, limit=100000)
embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
#embeddings = gensim.models.fasttext.load_facebook_vectors('weights/embeddings/cc.fi.300.bin')
print('Done...')


# generate index dictionaries
#char2idx, idx2char = prepare_data.encode_data(target_train)

# generate index dictionaries
#with open('weights/char2idx.pkl', 'wb') as f:
#    pickle.dump(char2idx, f, protocol=pickle.HIGHEST_PROTOCOL)

#with open('weights/idx2char.pkl', 'wb') as f:
#    pickle.dump(idx2char, f, protocol=pickle.HIGHEST_PROTOCOL)


# used for normalized
tag2idx = {'O': 1, 'PER': 2, 'LOC': 3, 'ORG': 4}
idx2tag = {1: 'O', 2: 'PER', 3: 'LOC', 4: 'ORG'}


with open('weights/char2idx.pkl', 'rb') as f:
    char2idx = pickle.load(f)
with open('weights/idx2char.pkl', 'rb') as f:
    idx2char = pickle.load(f)


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
encoder = Encoder(features_train[0].size(1), encoder_hidden_size, encoder_layers, batch_size, device).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)

# initialize the Decoder
decoder = Decoder(embedding_dim_chars, encoder_hidden_size, attention_hidden_size, num_filters, len(char2idx)+1, decoder_layers, encoder_layers, batch_size, attention_type, device).to(device)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)

# initialize the DecoderNER
decoder_ner = DecoderNER(embedding_dim_words, decoder_ner_hidden_size, len(tag2idx)+1, device).to(device)
decoder_ner_optimizer = optim.Adam(decoder_ner.parameters(), lr=decoder_ner_lr)


#print(vgg_extractor)
print(encoder)
print(decoder)
print(decoder_ner)

#total_trainable_params_vgg = sum(p.numel() for p in vgg_extractor.parameters() if p.requires_grad)
total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_trainable_params_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
total_trainable_params_decoder_ner = sum(p.numel() for p in decoder_ner.parameters() if p.requires_grad)

print('The number of trainable parameters is: %d' % (total_trainable_params_encoder + total_trainable_params_decoder + total_trainable_params_decoder_ner))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load('weights/small_normalized/state_dict_34.pt')
    #encoder.load_state_dict(checkpoint['encoder'])
    #decoder.load_state_dict(checkpoint['decoder'])
    #decoder_ner.load_state_dict(checkpoint['decoder_ner'])
    #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    #decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
    #decoder_ner_optimizer.load_state_dict(checkpoint['decoder_ner_optimizer'])

    criterion = nn.NLLLoss(ignore_index=0, reduction='mean')
    train(pairs_batch_train, pairs_batch_dev, encoder, decoder, decoder_ner, encoder_optimizer, decoder_optimizer, decoder_ner_optimizer, criterion, batch_size, num_epochs, device)
else:
    checkpoint = torch.load('weights/whole_normalized/state_dict_32.pt', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    decoder_ner.load_state_dict(checkpoint['decoder_ner'])


batch_size = 1

pairs_batch_train = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions(encoder, decoder, decoder_ner, language_model, batch_size, idx2char, idx2tag, pairs_batch_train, MAX_LENGTH)


