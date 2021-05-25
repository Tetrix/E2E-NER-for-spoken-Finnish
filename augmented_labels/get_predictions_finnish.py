import torch
from torch.utils.data import DataLoader

from jiwer import wer
import numpy as np
import fasttext

import utils.prepare_data as prepare_data
from model import Encoder, Decoder
from config.config import *

import utils.beam_search_decoding as bsd
from utils.calculate_f1 import print_scores

import pickle
import operator




def get_predictions(encoder, decoder, batch_size, idx2char, idx2tag,test_data, MAX_LENGTH):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        all_predictions = []
        all_predictions_rescored = []
        all_labels = []

        
        for l, batch in enumerate(test_data):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths = batch
            pad_input_seqs, pad_target_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device)
            
            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))

            features = pad_input_seqs.permute(1, 0, 2)
            res = bsd.beam_decode(decoder, features, decoder_hidden, target_seq_lengths, idx2char, encoder_output)
             
            candidates = []
            beam_sentences = res[0][0]
            beam_scores = res[0][1]
            
            true_labels = [idx2char[l.item()] for l in pad_target_seqs if l.item() not in (1, 2)]
            true_labels = ''.join(true_labels)
            
            predictions = [idx2char[l.item()] for l in beam_sentences[0] if l.item() not in (1, 2)]
            predictions = ''.join(predictions)

            all_predictions.append(predictions)
            all_labels.append(true_labels)
     
            
            # print statistics
            #print('new')
            #print(true_labels)
            #print(predictions)
            #print(rescored_predictions)

        
        transcripts_predicted, tags_predicted = split_entities(all_predictions)
        transcripts_true, tags_true = split_entities(all_labels)


        print('Word error rate: ', wer(transcripts_true, transcripts_predicted) * 100)
        #print('Word error rate: ', wer(all_labels, all_predictions) * 100)
        #print('Word error rate: ', wer(all_labels, all_predictions_rescored) * 100)
        #print('Micro AVG F1')
        #print_scores(tags_predicted, conventional_tags)

        #
        #np.save('output/parliament/tags_predicted.npy', tags_predicted)
        #np.save('output/parliament/transcripts_predicted.npy', transcripts_predicted)
       
        #all_predictions = np.load('output/parliament/transcripts_predicted.npy')
        #tags_predicted = np.load('output/parliament/tags_predicted.npy', allow_pickle=True)
        #
        ## save ASR output
        #with open('output/parliament/asr_output.txt', 'w') as f:
        #    for i in range(len(all_predictions)):
        #        sentence = all_predictions[i].split()
        #        for j in range(len(sentence)):
        #            f.write(sentence[j] + '\n')
        #        f.write('\n')


        ## save ASR output for e2e evaluation
        #with open('output/parliament/e2e_asr_combined.txt', 'w') as f:
        #    for i in range(len(all_predictions)):
        #        f.write(all_predictions[i])
        #        f.write('\n')


       
        ## save NER output
        #with open('output/parliament/e2e_ner.txt', 'w') as f:
        #    for sent in tags_predicted:
        #        for ent in sent:
        #            f.write(ent.rstrip())
        #            f.write('\n')
        #        f.write('\n')
 

       
        


# extract the named entities
def split_entities(data):
    transcripts = []
    tags = []
    temp_transcripts = []
    temp_tags = []
        
    for sent in data:
        sent = sent.split()
        for word in sent:
            if word not in ['O', 'PER', 'ORG', 'LOC']:
                temp_transcripts.append(word)
            else:
                temp_tags.append(word)

        if len(temp_transcripts) > len(temp_tags):
            temp_tags.append('O')

        temp_transcripts = ' '.join(temp_transcripts)
        transcripts.append(temp_transcripts)
        tags.append(temp_tags)
    
        temp_transcripts = []
        temp_tags = []

    return transcripts, tags






if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # test Parliament
    features_test = prepare_data.load_features_combined('../../TSD/augmented_labels/data/normalized/features/test.npy')
    target_test = prepare_data.load_transcripts('../../TSD/augmented_labels/data/normalized/augmented/parliament/test.txt')
    
    # compare againt conventional NER
    #features_test = prepare_data.load_features_combined('../augmented_labels/data/normalized/features/test.npy')
    #target_test = prepare_data.load_transcripts('output/parliament/e2e_asr_combined.txt')
    #tags_test = prepare_data.load_tags('output/parliament/conventional_ner.txt')

    features_test = features_test[:50]
    target_test = target_test[:50]


    print('Loading embeddings...')
    embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
    print('Done...')


    tag2idx = {'O': 1, 'PER': 2, 'LOC': 3, 'ORG': 4}
    idx2tag = {1: 'O', 2: 'PER', 3: 'LOC', 4: 'ORG'}


    with open('weights/char2idx_augmented.pkl', 'rb') as f:
        char2idx = pickle.load(f)
    with open('weights/idx2char_augmented.pkl', 'rb') as f:
        idx2char = pickle.load(f)


    # convert labels to indices
    indexed_target_test = prepare_data.label_to_idx(target_test, char2idx)
    indexed_target_word_test = prepare_data.word_to_idx(target_test, embeddings)

    test_data = prepare_data.combine_data(features_test, indexed_target_test)


    # initialize the Encoder
    encoder = Encoder(features_test[0].size(1), encoder_hidden_size, encoder_layers, batch_size, device).to(device)

    # initialize the Decoder
    decoder = Decoder(embedding_dim_chars, encoder_hidden_size, attention_hidden_size, num_filters, len(char2idx)+1, decoder_layers, encoder_layers, batch_size, attention_type, device).to(device)


    # load the model
    checkpoint = torch.load('weights/parliament/state_dict_21.pt', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])



    # evaluate
    batch_size = 1

    pairs_batch_train = DataLoader(dataset=test_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


    get_predictions(encoder, decoder, batch_size, idx2char, idx2tag, pairs_batch_train, MAX_LENGTH)


