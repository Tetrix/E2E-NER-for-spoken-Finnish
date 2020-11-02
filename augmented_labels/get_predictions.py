import torch
from jiwer import wer
import numpy as np

import utils.beam_search_decoding as bsd
from utils.language_model.generate import generate
from utils.calculate_f1 import print_scores



def get_predictions(encoder, decoder, language_model, batch_size, idx2char, test_data, MAX_LENGTH):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')

    encoder.eval()
    decoder.eval()
    language_model.eval()

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
             
            # Language model rescoring
            candidates = []
            beam_sentences = res[0][0]
            beam_scores = res[0][1]
            #
            #for k in range(len(beam_sentences)):
            #    temp = []
            #    for char in beam_sentences[k]:
            #        if idx2char[char.item()] not in ['<sos>', '<eos>']:
            #            temp.append(idx2char[char.item()])
            #    temp = ''.join(temp)
            #    candidates.append((temp, beam_scores[k]))
           
            #ranked_candidates = {}
            #for c in candidates:
            #    score = c[1]
            #    sent = []
            #    ner_tags = []
            #    for word in c[0].split():
            #        if word in ['O', 'PER', 'LOC', 'ORG']:
            #            ner_tags.append(word)
            #        else:
            #            sent.append(word)
            #        
            #    sent = ' '.join(sent)
            #    
            #    candidate, probability = generate(language_model, sent, ner_tags, score, device)
            #    ranked_candidates[probability] = candidate
            # 
            #rescored_predictions = {k: ranked_candidates[k] for k in sorted(ranked_candidates, reverse=True)}
            #
            ##print('ranked')
            ##for key, value in rescored_predictions.items():
            ##    print(key, value)
            #
            #rescored_predictions = rescored_predictions.values()
            #rescored_predictions = iter(rescored_predictions)
            #rescored_predictions = next(rescored_predictions)
            #all_predictions_rescored.append(rescored_predictions)
            # end of rescoring 


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


        #np.save('tags_predicted_eng.npy', tags_predicted)
        #np.save('transcripts_predicted_eng.npy', transcripts_predicted)


        print('Word error rate: ', wer(transcripts_true, transcripts_predicted) * 100)
        #print('Word error rate: ', wer(all_labels, all_predictions) * 100)
        #print('Word error rate: ', wer(all_labels, all_predictions_rescored) * 100)

       

        #transcripts_predicted = np.load('transcripts_predicted_eng.npy')
        #tags_predicted = np.load('tags_predicted_eng.npy', allow_pickle=True)


        # save ASR output
        #with open('output/e2e_asr.txt', 'w') as f:
        #    for i in range(len(transcripts_predicted)):
        #        sentence = transcripts_predicted[i].split()
        #        for j in range(len(sentence)):
        #            f.write(sentence[j] + '\n')
        #        f.write('\n')


        
        # load conventional ner and compare with e2e
        #with open('output/conventional_ner.txt', 'r') as f:
        #    data = f.readlines()
        #conventional_tags = []
        #temp = []
        #for i in data:
        #    if i != '\n':
        #        temp.append(i.rstrip())
        #    else:
        #        conventional_tags.append(temp)
        #        temp = []
        #
        #for i in range(len(conventional_tags)):
        #    if len(conventional_tags[i]) > len(tags_predicted[i]):
        #        tags_predicted[i].append('O')


        #print('Micro AVG F1')
        #print_scores(tags_predicted, conventional_tags)




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




            
            
