import torch
from jiwer import wer

import utils.beam_search_decoding as bsd



def get_predictions(encoder, decoder, decoder_ner, batch_size, idx2char, idx2tag, test_data, MAX_LENGTH):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')

    encoder.eval()
    decoder.eval()
    decoder_ner.eval()

    with torch.no_grad():
        all_predictions = []
        all_labels = []
        
        for l, batch in enumerate(test_data):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
            pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)

            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))

            # NER prediction
            decoder_ner_output, decoder_ner_hidden = decoder_ner(pad_word_seqs, encoder_hidden, word_seq_lengths)
            
            pad_tag_seqs = pad_tag_seqs.squeeze(-1)
            mask = pad_tag_seqs.clone()
            mask[mask != 0] = 1
            mask = mask.byte()
            ner_predictions = decoder_ner.crf.decode(decoder_ner_output, mask=mask)[0]
            # end sequence tagging

            features = pad_input_seqs.permute(1, 0, 2)
            res = bsd.beam_decode(decoder, features, decoder_hidden, target_seq_lengths, encoder_output)
            
            true_tags = [idx2tag[l.item()] for l in pad_tag_seqs]
            ner_predictions = [idx2tag[l] for l in ner_predictions if l != 0]

            #true_tags = true_tags[1:-1]
            #ner_predictions = ner_predictions[1:-1]

            true_labels = [idx2char[l.item()] for l in pad_target_seqs if l.item() not in (1, 2)]
            true_labels = ''.join(true_labels)
     
            predictions = [idx2char[l.item()] for l in res[0][0] if l.item() not in (1, 2)]
            predictions = ''.join(predictions)

            all_predictions.append(predictions)
            all_labels.append(true_labels)
    
            pad_tag_seqs = [i.item() for i in pad_tag_seqs]

            print('new')
            print(true_tags)
            print(ner_predictions)
            print(true_labels)
            print(predictions)
        print('Word error rate: ', wer(all_labels, all_predictions) * 100)



#def get_predictions(encoder, decoder, decoder_ner, batch_size, idx2char, idx2tag, test_data, MAX_LENGTH):
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    print('Evaluating...')
#   
#    encoder.eval()
#    decoder.eval()
#
#    with torch.no_grad():
#        all_predictions = []
#        all_labels = []
#
#        for l, batch in enumerate(test_data):
#            #all_predictions = []
#            #all_labels = []
#
#            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
#            pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)
#
#            predictions = []
#
#            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
#            #decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
#            decoder_hidden = encoder_hidden.sum(0, keepdim=True)
#            decoder_input = pad_target_seqs[0]
#            #decoder_context = torch.autograd.Variable(torch.zeros(1, batch_size, decoder.hidden_size*2)).to(device)
#
#            predictions.append(idx2char[decoder_input.item()])
#
#                
#            for i in range(MAX_LENGTH):
#                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
#                _, topi = decoder_output.topk(1)
#                decoder_input = topi.detach()
#
#                if decoder_input.item() != 0 and decoder_input.item() != 1:
#                    predictions.append(idx2char[decoder_input.item()])
#                
#                if decoder_input.item() == 2:
#                    break
#
#
#            true_labels = [idx2char[l.item()] for l in pad_target_seqs if l.item() != 0]
#            true_labels = ''.join(true_labels)
#            predictions = ''.join(predictions)
#
#            all_predictions.append(predictions)
#            all_labels.append(true_labels)
#
#            print('new')
#            print(true_labels)
#            print(predictions)
#            print('Word error rate: ', wer(true_labels, predictions) * 100) 
#        print('Word error rate: ', wer(all_labels, all_predictions) * 100)
#
#
#
