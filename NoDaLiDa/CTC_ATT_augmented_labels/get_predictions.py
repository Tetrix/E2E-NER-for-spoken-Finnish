import torch
from jiwer import wer
import numpy as np
import fastwer

import torch
import torch.nn.functional as F

from utils.calculate_f1 import print_scores
import utils.beam_search_decoding as bsd
#from utils.language_model_eng.generate import generate
#from utils.language_model_eng.new_lm.generate import generate

#from ctcdecode import CTCBeamDecoder

import operator

def get_predictions(encoder, decoder, language_model, batch_size, idx2char, test_data, MAX_LENGTH):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')

    encoder.eval()
    decoder.eval()
   
   
    with torch.no_grad():
        all_predictions = []
        all_predictions_ctc = []
        all_predictions_rescored = []
        all_labels = []
        all_tag_predictions = []
        all_tags = []
        partial_wer = []
        ctc_pred_new = []

        
        for l, batch in enumerate(test_data):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths = batch
            pad_input_seqs, pad_target_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device)
            
            encoder_output, encoder_hidden, encoder_output_prob = encoder(pad_input_seqs, input_seq_lengths)
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))

            attn_weights = F.softmax(torch.rand(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)
            

            features = pad_input_seqs.permute(1, 0, 2)
            res = bsd.beam_decode(decoder, features, decoder_hidden, target_seq_lengths, idx2char, attn_weights, encoder_output)
            
            candidates = []
            beam_sentences = res[0][0]
            beam_scores = res[0][1]


            ## rescoring
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
            #    #print(c[1], c[0])
            #    sent = []
            #    ner_tags = []
            #    for word in c[0].split():
            #        if word in ['O', 'PER', 'LOC', 'ORG']:
            #            ner_tags.append(word)
            #        else:
            #            sent.append(word)
            #        
            #    sent = ' '.join(sent)
            #    candidate, probability = generate(language_model, sent, score, device)
            #    ranked_candidates[probability] = candidate
            # 
            #rescored_predictions = {k: ranked_candidates[k] for k in sorted(ranked_candidates, reverse=True)}

            ##print('ranked')
            ##for key, value in rescored_predictions.items():
            ##    print(key, value)
            #
            #rescored_predictions = rescored_predictions.values()
            #rescored_predictions = iter(rescored_predictions)
            #rescored_predictions = next(rescored_predictions)
            #all_predictions_rescored.append(rescored_predictions)

            #print(all_predictions_rescored)
            # end of rescoring 



            true_labels = [idx2char[l.item()] for l in pad_target_seqs if l.item() not in (1, 2)]
            true_labels = ''.join(true_labels)
            
            predictions = [idx2char[l.item()] for l in beam_sentences[0] if l.item() not in (0, 1, 2)]
            predictions = ''.join(predictions)

            all_predictions.append(predictions)
            all_labels.append(true_labels)
            
     
            #print statistics
            #print('new')
            #print(true_labels)
            #print(predictions)
            #print(rescored_predictions)
        
        
        transcripts_predicted, tags_predicted = split_entities(all_predictions)
        transcripts_true, tags_true = split_entities(all_labels)

        print('Word error rate: ', wer(transcripts_true, transcripts_predicted) * 100)
        #print('Word error rate: ', wer(all_labels, all_predictions) * 100)
        #print('Word error rate: ', wer(all_labels, all_predictions_rescored) * 100)


        #np.save('output/english/finetuned/tags_predicted.npy', tags_predicted)
        #np.save('output/english/finetuned/transcripts_predicted.npy', transcripts_predicted)
       
        #all_predictions = np.load('output/english/finetuned/transcripts_predicted.npy')
        #tags_predicted = np.load('output/english/finetuned/tags_predicted.npy', allow_pickle=True)
        #
        ## save ASR output
        #with open('output/english/finetuned/asr_output.txt', 'w') as f:
        #    for i in range(len(all_predictions)):
        #        sentence = all_predictions[i].split()
        #        for j in range(len(sentence)):
        #            f.write(sentence[j] + '\n')
        #        f.write('\n')


        ## save ASR output for e2e evaluation
        #with open('output/english/finetuned/e2e_asr_combined.txt', 'w') as f:
        #    for i in range(len(all_predictions)):
        #        f.write(all_predictions[i])
        #        f.write('\n')


       
        ## save NER output
        #with open('output/english/finetuned/e2e_ner.txt', 'w') as f:
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




            
 
