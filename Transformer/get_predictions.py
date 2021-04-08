import torch
from jiwer import wer
import numpy as np
import fastwer

import torch
import torch.nn.functional as F

import utils.beam_search_decoding as bsd
from utils.calculate_f1 import print_scores
from utils.beam_search import Generator
#from utils.language_model.generate import generate
#from utils.language_model_eng.generate import generate_hypotheses

#from ctcdecode import CTCBeamDecoder

import operator

def get_predictions(transformer, batch_size, idx2char, idx2tag, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')

    transformer.eval()
   
   
    all_predictions = []
    all_labels = []
    all_tag_predictions = []
    all_tags = []

    generator = Generator(
                model=transformer,
                beam_size=5,
                max_seq_len=450,
                src_pad_idx=0,
                trg_pad_idx=0,
                trg_bos_idx=1,
                trg_eos_idx=2).to(device)


    for l, batch in enumerate(test_data):
        pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
        pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)

        predicted_indices = []

        #src_mask = transformer.generate_square_subsequent_mask(pad_input_seqs.size(0)).to(device)
        #src_key_padding_mask = transformer.make_len_mask(pad_input_seqs).byte().to(device)
        #src_key_padding_mask = src_key_padding_mask[:, :, 0].permute(1, 0)

        src_mask = transformer.generate_square_subsequent_mask(pad_input_seqs.size(0)).to(device)
        src_padding_mask = (pad_input_seqs == 0).transpose(0, 1)[:, :, 0].to(device)
        
        pred_seq = generator.generate_prediction(pad_input_seqs)
        
        prediction = [idx2char[l] for l in pred_seq if l not in [0, 1, 2]]
        prediction = ''.join(prediction)

        #memory = transformer.encoder(pad_input_seqs, mask=src_mask, src_key_padding_mask=src_padding_mask)
       
        #out_token = 1
        #predicted_indices = [out_token]

        #for i in range(300):
        #    tgt_tensor = torch.LongTensor([predicted_indices]).view(-1, 1).to(device)
        #    tgt_mask = transformer.generate_square_subsequent_mask(i + 1).to(device)
        #    
        #    #memory_mask = torch.zeros(tgt_tensor.shape[0], memory.shape[0]).to(device).type(torch.bool)

        #    decoder_output = transformer.decoder(
        #            tgt=tgt_tensor,
        #            memory=memory,
        #            memory_mask=None,
        #            tgt_mask=tgt_mask,
        #            memory_key_padding_mask=src_padding_mask,
        #            tgt_key_padding_mask=None)


        #    output = transformer.out(decoder_output)
        #    output = F.softmax(output, dim=-1)
        #    output = output[-1]
        #    topi, topk = output.max(dim=-1)
        #    out_token = topk.item()
        #    predicted_indices.append(out_token)
        #     
        #    if out_token == 2:
        #        break

        #prediction = [idx2char[l] for l in predicted_indices if l not in [0]]
        #prediction = ''.join(prediction)


        true_labels = [idx2char[l.item()] for l in pad_target_seqs if l.item() not in [1, 2]]
        true_labels = ''.join(true_labels)
        

        all_predictions.append(prediction)
        all_labels.append(true_labels)

        ## print statistics
        #print('new')
        #print(true_labels)
        #print(prediction)

    print('Word error rate: ', wer(all_labels, all_predictions) * 100)            
       
       
        #print('Word error rate: ', wer(all_labels, all_predictions) * 100)
        #print('Character error rate: ', fastwer.score(all_predictions, all_labels, char_level=True))
        #print('Micro AVG F1')
        #print_scores(all_tag_predictions, all_tags)
       


        #all_predictions = np.array(all_predictions)
        #all_labels = np.array(all_labels)
        
        #np.save('predictions_eng.npy', all_predictions)
        #np.save('true_eng.npy', all_labels)

       
        #all_predictions = np.load('predictions_libri_other.npy')
        #all_labels = np.load('true_libri_other.npy')

       
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



