import torch
from jiwer import wer

import utils.beam_search_decoding as bsd
from utils.language_model.generate import generate



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
            
            for k in range(len(beam_sentences)):
                temp = []
                for char in beam_sentences[k]:
                    if idx2char[char.item()] not in ['<sos>', '<eos>']:
                        temp.append(idx2char[char.item()])
                temp = ''.join(temp)
                candidates.append((temp, beam_scores[k]))
            

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


            true_labels = [idx2char[l.item()] for l in pad_target_seqs if l.item() not in (1, 2)]
            true_labels = ''.join(true_labels)
            
            predictions = [idx2char[l.item()] for l in beam_sentences[0] if l.item() not in (1, 2)]
            predictions = ''.join(predictions)

            all_predictions.append(predictions)
            all_labels.append(true_labels)
     
            
            # print statistics
            print('new')
            print(true_labels)
            print(predictions)
            #print(rescored_predictions)
        print('Word error rate: ', wer(all_labels, all_predictions) * 100)
        print('Word error rate: ', wer(all_labels, all_predictions_rescored) * 100)


