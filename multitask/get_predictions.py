import torch
from jiwer import wer
import numpy as np

import utils.beam_search_decoding as bsd
from utils.calculate_f1 import print_scores
from utils.language_model.generate import generate

import operator

def get_predictions(encoder, decoder, decoder_ner, language_model, batch_size, idx2char, idx2tag, test_data, MAX_LENGTH):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')

    encoder.eval()
    decoder.eval()
    decoder_ner.eval()
    language_model.eval()

    with torch.no_grad():
        all_predictions = []
        all_predictions_rescored = []
        all_labels = []
        all_tag_predictions = []
        all_tags = []
        partial_wer = []

        
        for l, batch in enumerate(test_data):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
            pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)
            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))

            # NER prediction
            encoder_hidden_ner = (encoder_hidden[0][:2, :, :], encoder_hidden[1][:2, :, :])
            decoder_ner_output, decoder_ner_hidden = decoder_ner(pad_word_seqs, encoder_hidden_ner, word_seq_lengths, encoder_output)

            ner_predictions = decoder_ner.crf.decode(decoder_ner_output)[0]
            # end sequence tagging

            features = pad_input_seqs.permute(1, 0, 2)
            res = bsd.beam_decode(decoder, features, decoder_hidden, target_seq_lengths, idx2char, encoder_output)
            
            # Language model rescoring
            candidates = []
            beam_sentences = res[0][0]
            beam_scores = res[0][1]
            
            for k in range(len(beam_sentences)):
                temp = []
                for char in beam_sentences[k]:
                    if idx2char[char.item()] not in ['<sos>', '<eos>']:
                        temp.append(idx2char[char.item()])
                temp = ''.join(temp)
                candidates.append((temp, beam_scores[k]))

            #candidates.append(('blabla ajsjsjs ahahshdasdada sjkhfjskdhhkjsds ffshkfhkj sdhjk fsdhjk f hjksdf hjksd', 0))

            ranked_candidates = {}
            for c in candidates:
                candidate, probability = generate(language_model, c[0], c[1], device)
                ranked_candidates[probability] = candidate
             
            rescored_predictions = {k: ranked_candidates[k] for k in sorted(ranked_candidates, reverse=True)}
            
            #print('ranked')
            #for key, value in rescored_predictions.items():
            #    print(key, value) 
            
            rescored_predictions = rescored_predictions.values()
            rescored_predictions = iter(rescored_predictions)
            rescored_predictions = next(rescored_predictions)
            all_predictions_rescored.append(rescored_predictions)
            # end of rescoring 

            true_tags = [idx2tag[l.item()] for l in pad_tag_seqs]
            ner_predictions = [idx2tag[l] for l in ner_predictions if l != 0]
            
            all_tag_predictions.append(ner_predictions)
            all_tags.append(true_tags)
            

            true_labels = [idx2char[l.item()] for l in pad_target_seqs if l.item() not in (1, 2)]
            true_labels = ''.join(true_labels)
            
            predictions = [idx2char[l.item()] for l in beam_sentences[0] if l.item() not in (1, 2)]
            predictions = ''.join(predictions)

            all_predictions.append(predictions)
            all_labels.append(true_labels)
            
            
           
            # print statistics
            #print('new')
            #print(true_tags)
            #print(ner_predictions)
            #print(true_labels)
            #print(predictions)
            #print(rescored_predictions)
        
       
        print('Word error rate: ', wer(all_labels, all_predictions) * 100)
        print('Word error rate: ', wer(all_labels, all_predictions_rescored) * 100)
        print('Micro AVG F1')
        print_scores(all_tag_predictions, all_tags)
       

        #all_predictions = np.array(all_predictions)
        #all_labels = np.array(all_labels)
        
        #np.save('predictions.npy', all_predictions)
        #np.save('true.npy', all_labels)
        
       
        #all_predictions = np.load('predictions.npy')
        #all_labels = np.load('true.npy')

       
        ## save ASR output
        #with open('output/asr_output.txt', 'w') as f:
        #    for i in range(len(all_predictions)):
        #        sentence = all_predictions[i].split()
        #        for j in range(len(sentence)):
        #            f.write(sentence[j] + '\n')
        #        f.write('\n')


        ## save ASR output for e2e evaluation
        #with open('output/e2e_asr_combined.txt', 'w') as f:
        #    for i in range(len(all_predictions)):
        #        f.write(all_predictions[i])
        #        f.write('\n')


       
        ## save NER output
        #with open('output/e2e_ner.txt', 'w') as f:
        #    for i in range(len(all_tag_predictions)):
        #        sentence = all_labels[i].split()
        #        for j in range(len(all_tag_predictions[i])):
        #            f.write(sentence[j] + '\t' + all_tag_predictions[i][j] + '\n')
        #        f.write('\n') 



