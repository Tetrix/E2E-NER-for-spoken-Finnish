import numpy as np
import os

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence


def load_features(features_path):
    feature_array = []
    for file in sorted(os.listdir(features_path)):
        feature = np.load(os.path.join(features_path, file))
        feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    return feature_array

#encoding='ISO-8859-1')
def load_labels(features_path):
    label_array = []
    for file in sorted(os.listdir(features_path)):
        with open(os.path.join(features_path, file), 'r', encoding='utf-8') as f:
            transcript = f.readlines()
            label_array.append(transcript)

    return label_array



#def encode_data(labels_data):
#    char2idx = {}
#    idx2char = {}
#
#
#    char2idx['<sos>'] = 1
#    idx2char[1] = '<sos>'
#
#    char2idx['<eos>'] = 2
#    idx2char[2] = '<eos>'
#
#    char2idx['<UNK>'] = 3
#    idx2char[3] = '<UNK>'
#
#
#    #char2idx['<O>'] = 4
#    #idx2char[4] = '<O>'
#    #char2idx['<PER>'] = 5
#    #idx2char[5] = '<PER>'
#
#
#    for sent in labels_data:
#        for char in sent[0]:
#            if char not in char2idx:
#                char2idx[char] = len(char2idx) + 1
#                idx2char[len(idx2char) + 1] = char
#    
#    return char2idx, idx2char
#

def encode_data(labels_data):
    char2idx = {}
    idx2char = {}


    char2idx['<sos>'] = 1
    idx2char[1] = '<sos>'

    char2idx['<eos>'] = 2
    idx2char[2] = '<eos>'

    char2idx['<UNK>'] = 3
    idx2char[3] = '<UNK>'
    
    char2idx['O'] = 4
    idx2char[4] = 'O'

    char2idx['PER'] = 5
    idx2char[5] = 'PER'
    
    char2idx['LOC'] = 6
    idx2char[6] = 'LOC'
    
    char2idx[' '] = 7
    idx2char[7] = ' '


    for sent in labels_data:
        sentence = sent[0].split()
        for word in sentence:
            if word in ['<UNK>', '<sos>', '<eos>', 'O', 'PER', 'LOC', ' ']:
                pass
            else:
                for char in word:
                    if char not in char2idx:
                        char2idx[char] = len(char2idx) + 1
                        idx2char[len(idx2char) + 1] = char
    
    return char2idx, idx2char



#def label_to_idx(labels, char2idx):
#    res = []
#    for sent in labels:
#        temp_sent = [] 
#        for char in sent[0]:
#            temp_sent.append([char2idx[char]])
#        # add eos
#        temp_sent.append([char2idx[' ']])
#        temp_sent.append([char2idx['<eos>']])
#
#        #res.append(autograd.Variable(torch.LongTensor(temp_sent)))
#        res.append(torch.LongTensor(temp_sent))
#
#    return res


def label_to_idx(labels, char2idx):
    res = []
    for sent in labels:
        temp_sent = [] 
        sent = sent[0].split()
        for i, word in enumerate(sent):
            if word in ['<UNK>', '<sos>', '<eos>', 'O', 'PER', 'LOC']:
                temp_sent.append([char2idx[word]])
            else:
                for char in word:
                    temp_sent.append([char2idx[char]])
            if len(sent) != (i):
                temp_sent.append([char2idx[' ']])

        # add eos
        temp_sent.append([char2idx[' ']])
        temp_sent.append([char2idx['<eos>']])
        res.append(torch.LongTensor(temp_sent))

    return res




# extra data to be removed so that it can be divided in equal batches
def remove_extra(data, batch_size):
    extra = len(data) % batch_size
    if extra != 0:
        data = data[:-extra][:]
    return data


def combine_data(features, indexed_labels):
    res = []

    for i in range(len(features)):
        res.append((features[i], indexed_labels[i]))

    return res


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, output_seqs = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]
    output_seq_lengths = [len(seq) for seq in output_seqs]


    # pad input sequences
    padding_value = 0
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)


    # pad output sequences
    pad_output_seqs = []
    for i in output_seqs:
        padded = i.new_zeros(max(output_seq_lengths) - i.size(0))
        # padded[:] = 99
        pad_output_seqs.append(torch.cat((i, padded.view(-1, 1)), dim=0))

    pad_output_seqs = torch.stack(pad_output_seqs)
    pad_output_seqs = pad_output_seqs.permute(1, 0, 2)


    return pad_input_seqs, input_seq_lengths, pad_output_seqs, output_seq_lengths
