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

print('libri_asr_big_new clean 11')

print(device)

# load language model related stuff
language_model = CharRNN(n_characters, 1024, n_characters, 'lstm', 2)
language_model.load_state_dict(torch.load('utils/language_model/lm_weights/language_model_1024_10000.pt', map_location='cpu'))
language_model = language_model.to(device)
language_model.eval()


# load features and labels
print('Loading data..')

# whole data ASR
#features_train = prepare_data.load_features('data/normalized/features/train')
#target_train = prepare_data.load_transcripts('data/normalized/transcripts/train.txt')

#features_dev = prepare_data.load_features_combined('data/normalized/features/dev.npy')
#target_dev = prepare_data.load_transcripts('data/normalized/transcripts/dev.txt')


# LibriSpeech data ASR
#features_train = prepare_data.load_features('data/normalized/features/libri/train')
#target_train = prepare_data.load_transcripts('data/normalized/transcripts/libri/train.txt')

#features_dev = prepare_data.load_features_combined('data/normalized/features/libri/dev.npy')
#target_dev = prepare_data.load_transcripts('data/normalized/transcripts/libri/dev.txt')



# whole data LibriSpeech
#features_train = prepare_data.load_features('data/normalized/features/libri/train')
#target_train = prepare_data.load_transcripts('data/normalized/augmented/libri_train.txt')

#features_dev = prepare_data.load_features_combined('data/normalized/features/libri/dev.npy')
#target_dev = prepare_data.load_transcripts('data/normalized/augmented/libri_dev.txt')


# whole data normalized augmented
#features_train = prepare_data.load_features('data/normalized/features/train_small')
#target_train = prepare_data.load_transcripts('data/normalized/augmented/train_small.txt')

#features_dev = prepare_data.load_features_combined('data/normalized/features/dev_small.npy')
#target_dev = prepare_data.load_transcripts('data/normalized/augmented/dev_small.txt')



# test normalized ASR
#features_train = prepare_data.load_features_combined('data/normalized/features/test.npy')
#target_train = prepare_data.load_transcripts('data/normalized/transcripts/test.txt')


# test normalized augmented
#features_train = prepare_data.load_features_combined('data/normalized/features/test.npy')
#target_train = prepare_data.load_transcripts('data/normalized/augmented/test.txt')


# test LibriSpeech
#features_train = prepare_data.load_features_combined('data/normalized/features/libri/test_clean.npy')
#target_train = prepare_data.load_transcripts('data/normalized/augmented/libri_test_clean.txt')


# test LibriSpeech ASR
features_train = prepare_data.load_features_combined('data/normalized/features/libri/test_clean.npy')
target_train = prepare_data.load_transcripts('data/normalized/transcripts/libri/test_clean.txt')


# test English out-of-domain
#features_train = prepare_data.load_features_combined('data/normalized/features/eng_test/test_eng.npy')
#target_train = prepare_data.load_transcripts('data/normalized/augmented/eng_test/test_eng.txt')


# test English out-of-domain ASR
#features_train = prepare_data.load_features_combined('data/normalized/features/eng_test/test_eng.npy')
#target_train = prepare_data.load_transcripts('data/normalized/transcripts/eng_test/test_eng.txt')


#features_train = features_train[:10]
#target_train = target_train[:10]

features_dev = features_train
target_dev = target_train

print('Done...')


print('Loading embeddings...')
#embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
embeddings = fasttext.load_model('weights/embeddings/crawl-300d-2M-subword.bin')
#embeddings = gensim.models.fasttext.load_facebook_vectors('weights/embeddings/cc.fi.300.bin')
print('Done...')



with open('weights/char2idx_libri.pkl', 'rb') as f:
    char2idx = pickle.load(f)
with open('weights/idx2char_libri.pkl', 'rb') as f:
    idx2char = pickle.load(f)


# convert labels to indices
indexed_target_train = prepare_data.label_to_idx(target_train, char2idx)
indexed_target_dev = prepare_data.label_to_idx(target_dev, char2idx)


# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train, indexed_target_train)
dev_data = prepare_data.combine_data(features_dev, indexed_target_dev)


# use subsample of the data
#train_data = train_data[:5]
#dev_data = dev_data[:5]
# test_data = dev_data[:100]


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


print(encoder)
print(decoder)

#total_trainable_params_vgg = sum(p.numel() for p in vgg_extractor.parameters() if p.requires_grad)
total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_trainable_params_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

print('The number of trainable parameters is: %d' % (total_trainable_params_encoder + total_trainable_params_decoder))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load('weights/libri_asr_big/state_dict_22.pt')
    #encoder.load_state_dict(checkpoint['encoder'])
    #decoder.load_state_dict(checkpoint['decoder'])
    #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    #decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    criterion = nn.NLLLoss(ignore_index=0, reduction='mean')
    train(pairs_batch_train, pairs_batch_dev, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, num_epochs, device)
else:
    checkpoint = torch.load('weights/libri_asr_big_new/state_dict_11.pt', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])


batch_size = 1
dev_data = dev_data[:]
#train_data = train_data[:64]

pairs_batch_train = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions(encoder, decoder, language_model, batch_size, idx2char, pairs_batch_train, MAX_LENGTH)


