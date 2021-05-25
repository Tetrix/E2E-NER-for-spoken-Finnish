import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

import numpy as np
import pickle
import fasttext

import utils.prepare_data as prepare_data
from model import Encoder, Decoder
from config.config import *
from train import train
from get_predictions import get_predictions



torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(device)


# load features and labels
print('Loading data..')

# Parliament data ASR
#features_train = prepare_data.load_features('data/normalized/features/train')
#target_train = prepare_data.load_transcripts('data/normalized/transcripts/train.txt')

#features_dev = prepare_data.load_features_combined('data/normalized/features/dev.npy')
#target_dev = prepare_data.load_transcripts('data/normalized/transcripts/dev.txt')



# Parlaiament data augmented
features_train = prepare_data.load_features('data/normalized/features/train')
target_train = prepare_data.load_transcripts('data/normalized/augmented/parliament/train.txt')

features_dev = prepare_data.load_features_combined('data/normalized/features/dev.npy')
target_dev = prepare_data.load_transcripts('data/normalized/augmented/parliament/dev.txt')


print('Done...')


print('Loading embeddings...')
embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
print('Done...')



with open('weights/char2idx_augmented.pkl', 'rb') as f:
    char2idx = pickle.load(f)
with open('weights/idx2char_augmented.pkl', 'rb') as f:
    idx2char = pickle.load(f)


# convert labels to indices
indexed_target_train = prepare_data.label_to_idx(target_train, char2idx)
indexed_target_dev = prepare_data.label_to_idx(target_dev, char2idx)


# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train, indexed_target_train)
dev_data = prepare_data.combine_data(features_dev, indexed_target_dev)



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
    checkpoint = torch.load('weights/parliament/state_dict_21.pt')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    criterion = nn.NLLLoss(ignore_index=0, reduction='mean')
    train(pairs_batch_train, pairs_batch_dev, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, num_epochs, device)
else:
    checkpoint = torch.load('weights/parliament_asr/state_dict_24.pt', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
