import numpy as np
import os

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence



def load_features_combined(features_path):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)
    
    for feature in features:
        feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    return feature_array


def load_features(features_path):
    feature_array = []
    for file in sorted(os.listdir(features_path)):
        feature = np.load(os.path.join(features_path, file))
        feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    return feature_array


def load_transcripts(features_path):
    label_array = []
    with open(features_path, 'r') as f:
        data = f.readlines()

    for sent in data:
        sent = sent.rstrip()
        label_array.append([sent])

    return label_array


def load_labels(features_path):
    label_array = []
    for file in sorted(os.listdir(features_path)):
        if os.path.isfile(os.path.join(features_path, file)):
            with open(os.path.join(features_path, file), 'r', encoding='utf-8') as f:
                transcript = f.readlines()
                label_array.append(transcript)

    return label_array


#TODO Add <UNK> as a special token. Currently it is processed as characters
def encode_data(labels_data):
    char2idx = {}
    idx2char = {}


    char2idx['<sos>'] = 1
    idx2char[1] = '<sos>'

    char2idx['<eos>'] = 2
    idx2char[2] = '<eos>'

    char2idx['<UNK>'] = 3
    idx2char[3] = '<UNK>'


    for sent in labels_data:
        sentence = sent[0].split()
        for char in sent[0]:
            if char not in char2idx:
                char2idx[char] = len(char2idx) + 1
                idx2char[len(idx2char) + 1] = char

    return char2idx, idx2char


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
#    for sent in labels_data:
#        sentence = sent[0].split()
#        for word in sentence:
#            if word in ['<UNK>', '<sos>', '<eos>']:
#                char2idx[word] = len(char2idx) + 1
#                idx2char[len(idx2char) + 1] = char
#            else:
#                for char in word:
#                    if char not in char2idx:
#                        char2idx[char] = len(char2idx) + 1
#                        idx2char[len(idx2char) + 1] = char
#
#    return char2idx, idx2char


def label_to_idx(labels, char2idx):
    res = []
    for sent in labels:
        temp_sent = []
        for char in sent[0]:
            temp_sent.append([char2idx[char]])
        # add eos token
        temp_sent.append([char2idx[' ']])
        temp_sent.append([char2idx['<eos>']])

        #res.append(autograd.Variable(torch.LongTensor(temp_sent)))
        res.append(torch.LongTensor(temp_sent))
    return res


def prepare_word_sequence(seq, embeddings):
    res = []
    seq = seq.split()
    for w in seq:
        try:
            res.append(embeddings[w])
        except:
            res.append(np.random.normal(scale=0.6, size=(300, )))
    res = autograd.Variable(torch.FloatTensor(res))
 
    return res


def word_to_idx(data, embeddings):
    res = []
    for seq in range(len(data)):
        res.append(prepare_word_sequence(data[seq][0], embeddings))
    return res



def tag_to_idx(tags, tag2idx):
    res = []
    for sent in tags:
        temp_sent = []
        for tag in sent:
            if tag != '\n':
                tag = tag.rstrip()
                temp_sent.append([tag2idx[tag]])
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

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)

    # pad output sequences
    pad_output_seqs = pad_sequence(output_seqs, padding_value=padding_value)

    return pad_input_seqs, input_seq_lengths, pad_output_seqs, output_seq_lengths






