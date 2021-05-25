import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle
import gensim
import fasttext
import time

import utils.prepare_data as prepare_data
from utils.radam import RAdam
from model import Encoder, Decoder
from config.config import *
from train import train
from get_predictions import get_predictions


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#torch.autograd.set_detect_anomaly(True)

# load language model related stuff
#from utils.language_model_eng.model import CharRNN
#from utils.language_model_eng.helpers import *
#
#language_model = CharRNN(n_characters, 1024, n_characters, 'lstm', 2)
#language_model.load_state_dict(torch.load('utils/language_model_eng/lm_weights/language_model_eng.pt', map_location='cpu'))
#language_model = language_model.to(device)
#language_model.eval()

#from utils.language_model_eng.new_lm.model import CharRNN
#with open('utils/language_model_eng/new_lm/data/whole.txt', 'r') as f:
#    text = f.read()
#chars = tuple(set(text))
#int2char = dict(enumerate(chars))
#char2int = {ch: ii for ii, ch in int2char.items()}
## Define and print the net
#
#language_model = CharRNN(chars, 1024, 3).to(device)
#language_model.load_state_dict(torch.load('utils/language_model_eng/new_lm/models/1024_3_20.pt', map_location='cpu'))
#language_model.eval()
language_model = []


start_time = time.time()

print('libri_asr_temp clean 17')
print(device)

# load features and labels
print('Loading data..')


# Parliament data
#features_train = prepare_data.load_features('../augmented_labels/data/normalized/features/train')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/parliament/train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/parliament/dev.txt')


# Parliament ASR
#features_train = prepare_data.load_features('data/normalized/features/train')
#target_train = prepare_data.load_transcripts('data/normalized/transcripts/train.txt')

#features_dev = prepare_data.load_features_combined('data/normalized/features/dev.npy')
#target_dev = prepare_data.load_transcripts('data/normalized/transcripts/dev.txt')


# English-Gold data
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/train.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/eng_ood/train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/eng_ood/dev.txt')


# English-Gold ASR data
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/train.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/eng_ood/train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/eng_ood/dev.txt')



# Swedish data
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/train.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/swedish/train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/swedish/dev.txt')


# Swedish ASR
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/train.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/swedish/train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/swedish/dev.txt')



# LibriSpeech ASR data
#features_train = prepare_data.load_features('../augmented_labels/data/normalized/features/libri/train')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/libri/train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/libri/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/libri/dev.txt')




# LibriSpeech data
#features_train = prepare_data.load_features('../augmented_labels/data/normalized/features/libri/train')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/libri/train.txt')

#features_dev = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/libri/dev.npy')
#target_dev = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/libri/dev.txt')



# test Parliament
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/test.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/test.txt')


# test Swedish
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/test.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/swedish/test.txt')


# test Swedish ASR
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/swedish/test.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/swedish/test.txt')


# test English-Gold data
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/test.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/eng_ood/test.txt')


# test English-Gold ASR data
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/test.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/eng_ood/test.txt')




# test LibriSpeech
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/libri/test_other.npy')
#target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/augmented/libri/test_other.txt')


# test LibriSpeech ASR
features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/libri/test_clean.npy')
target_train = prepare_data.load_transcripts('../augmented_labels/data/normalized/transcripts/libri/test_clean.txt')



# For the E2E evaluation
#features_train = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/eng_ood/test.npy')
#target_train = prepare_data.load_transcripts('output/english/e2e_asr_combined.txt')
#tags_train = prepare_data.load_tags('output/english/conventional_ner.txt')
#tags_predicted = prepare_data.load_tags('output/english/e2e_ner.txt')


features_train = features_train[:100]
target_train = target_train[:100]
#tags_train = tags_train[:200]
#tags_predicted = tags_predicted[:200]

#all_tag_predictions = np.array(tags_predicted)
#all_tags = np.array(tags_train)
#np.save('plotting/english/predictions.npy', all_tag_predictions)
#np.save('plotting/english/true.npy', all_tags)


features_dev = features_train
target_dev = target_train
#tags_dev = tags_train

#tags_predicted_new = []
#tags_train_new = []
#for i in range(len(tags_train)):
#    if len(tags_train[i]) == len(tags_predicted[i]):
#        tags_predicted_new.append(tags_predicted[i])
#        tags_train_new.append(tags_train[i])


# evaluate NER on generated transcripts
#from utils.calculate_f1 import print_scores
#print('Micro AVG F1')
#print_scores(tags_predicted_new, tags_train_new)
       


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
encoder = Encoder(features_train[0].size(1), encoder_hidden_size, encoder_layers, len(char2idx_ctc), batch_size, device).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)

# initialize the Decoder
decoder = Decoder(embedding_dim_chars, encoder_hidden_size, attention_hidden_size, num_filters, len(char2idx)+1, decoder_layers, encoder_layers, batch_size, attention_type, device).to(device)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)

print(encoder)
print(decoder)

total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_trainable_params_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

print('The number of trainable parameters is: %d' % (total_trainable_params_encoder + total_trainable_params_decoder))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    checkpoint = torch.load('weights/english_asr_finetuned/state_dict_1.pt')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    criterion = nn.NLLLoss(ignore_index=0, reduction='mean')
    ctc_loss = torch.nn.CTCLoss(blank=char2idx_ctc['~'], reduction='mean', zero_infinity=True)
    train(pairs_batch_train, 
            pairs_batch_dev, 
            encoder, 
            decoder,
            encoder_optimizer, 
            decoder_optimizer, 
            criterion, 
            ctc_loss, 
            batch_size, 
            num_epochs, 
            device,
            len(train_data),
            len(dev_data))
else:
    checkpoint = torch.load('weights/libri_asr/state_dict_17.pt', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])


batch_size = 1

pairs_batch_train = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions(encoder, decoder, language_model, batch_size, idx2char, pairs_batch_train, MAX_LENGTH)
print("--- %s seconds ---" % (time.time() - start_time))
