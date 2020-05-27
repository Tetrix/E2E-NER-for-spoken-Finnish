import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle

import utils.prepare_data as prepare_data
from utils.radam import RAdam
from model import Encoder, Decoder
from config.config import *
from train import train
from get_predictions import get_predictions



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# load features and labels
#target_whole = prepare_data.load_labels('data/transcripts/whole_small')

print('Loading data..')
#features_train = prepare_data.load_features('/m/triton/scratch/work/porjazd1/E2E_NER/E2E-NER/data/features/dev_subsample')
#target_train = prepare_data.load_labels('/m/triton/scratch/work/porjazd1/E2E_NER/E2E-NER/data/transcripts/dev_subsample')

#features_train = features_train[:20]
#target_train = target_train[:20]

features_train = prepare_data.load_features('data/features/dev_subsample')
target_train = prepare_data.load_labels('data/transcripts/dev_subsample')

features_dev = features_train
target_dev = target_train

#features_dev = prepare_data.load_features('data/features/dev_small')
#target_dev = prepare_data.load_labels('data/transcripts/dev_small')


print('Done...')

#features_test = prepare_data.load_features('data/features/test')
#target_test = prepare_data.load_labels('data/transcripts/test')

# generate index dictionaries
#char2idx, idx2char = prepare_data.encode_data(target_train + target_dev)
#with open('weights/char2idx_new.pkl', 'wb') as f:
#    pickle.dump(char2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
#with open('weights/idx2char_new.pkl', 'wb') as f:
#    pickle.dump(idx2char, f, protocol=pickle.HIGHEST_PROTOCOL)



with open('weights/char2idx_new.pkl', 'rb') as f:
    char2idx = pickle.load(f)
with open('weights/idx2char_new.pkl', 'rb') as f:
    idx2char = pickle.load(f)

# convert labels to indices
indexed_target_train = prepare_data.label_to_idx(target_train, char2idx)
indexed_target_dev = prepare_data.label_to_idx(target_dev, char2idx)
# indexed_target_test = prepare_data.label_to_idx(target_test, char2idx)


# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train, indexed_target_train)
dev_data = prepare_data.combine_data(features_dev, indexed_target_dev)
# test_data = prepare_data.combine_data(features_test, indexed_target_test)


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
#encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr_rate)
encoder_optimizer = optim.Adam(encoder.parameters())

# initialize the Decoder
decoder = Decoder(embedding_dim, encoder_hidden_size, len(char2idx)+1, decoder_layers, encoder_layers, device).to(device)
#decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr_rate)
decoder_optimizer = optim.Adam(decoder.parameters())

print(encoder)
print(decoder)

total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_trainable_params_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print('The number of trainable parameters is: %d' % (total_trainable_params_encoder + total_trainable_params_decoder))

# train
if skip_training == False:
    #load weights to continue training from a checkpoint
    checkpoint = torch.load('weights/new/state_dict_55.pt')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    criterion = nn.NLLLoss(ignore_index=0, reduction='mean')
    train(pairs_batch_train, pairs_batch_dev, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, num_epochs, device)
else:
    checkpoint = torch.load('weights/new/state_dict_67.pt')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])


batch_size = 1
dev_data = dev_data[:1000]
#train_data = train_data[:64]

pairs_batch_train = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)

encoder.eval()
decoder.eval()

get_predictions(encoder, decoder, batch_size, idx2char, pairs_batch_train, MAX_LENGTH)


