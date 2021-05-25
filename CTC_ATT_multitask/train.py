import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.early_stopping import EarlyStopping


def train(pairs_batch_train, pairs_batch_dev, encoder, decoder, decoder_ner, encoder_optimizer, decoder_optimizer, decoder_ner_optimizer, criterion, ctc_loss, batch_size, num_epochs, device, train_data_len, dev_data_len):
    clip = 1.0
    tf_rate = 1
    lambda_factor = 0.8
    ctc_weight = 0.2

    for epoch in range(10):
        encoder.train()
        decoder.train()
        decoder_ner.train()
        
        batch_loss_train = 0
        batch_loss_dev = 0

        for iteration, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
            pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)
       
            train_loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            decoder_ner_optimizer.zero_grad()

            encoder_output, encoder_hidden, encoder_output_prob = encoder(pad_input_seqs, input_seq_lengths)
            decoder_input = torch.ones(batch_size, 1).long().to(device)
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))

            teacher_forcing = True if random.random() <= tf_rate else False
           
            attn_weights = F.softmax(torch.ones(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)

            if teacher_forcing:
                for i in range(0, pad_target_seqs.size(0)):
                    decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, attn_weights)
                    target = pad_target_seqs.squeeze()
                    train_loss += criterion(decoder_output, target[i])
                    decoder_input = pad_target_seqs[i].detach()
            else:
                for i in range(0, pad_target_seqs.size(0)):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                    _, topi = decoder_output.topk(1)
                    target = pad_target_seqs.squeeze()
                    train_loss += criterion(decoder_output, target[i])
                    decoder_input = topi.detach()
           
           
             CTC LOSS
            targets = pad_target_seqs.squeeze().permute(1, 0)
            input_lengths = torch.ones(encoder_output_prob.size(1)) * encoder_output_prob.size(0)
            input_lengths = input_lengths.type(torch.LongTensor)
            target_seq_lengths = np.array(target_seq_lengths)
            target_lengths = torch.from_numpy(target_seq_lengths)
            
            train_loss_ctc = ctc_loss(encoder_output_prob, targets, input_lengths, target_lengths)
            asr_loss = ((1 - ctc_weight) * train_loss) + (ctc_weight * train_loss_ctc)
            

            # NER BRANCH
            encoder_hidden_ner = (encoder_hidden[0][:2, :, :], encoder_hidden[1][:2, :, :])
            decoder_ner_output, decoder_ner_hidden = decoder_ner(pad_word_seqs, encoder_hidden_ner, word_seq_lengths, encoder_output)
            pad_tag_seqs = pad_tag_seqs.squeeze()
            
            mask = pad_tag_seqs.clone()
            mask[mask != 0] = 1
            mask = mask.byte()

            negative_log_likelihood = -decoder_ner.crf(decoder_ner_output, pad_tag_seqs, mask=mask)

            # interpolate the loss
            loss = (lambda_factor * asr_loss) + ((1 - lambda_factor) * negative_log_likelihood.item())
            
            batch_loss_train += loss.detach()
            
            # backward step
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder_ner.parameters(), clip)

            encoder_optimizer.step()
            decoder_optimizer.step()
            decoder_ner_optimizer.step()


        # CALCULATE EVALUATION
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            decoder_ner.eval()

            for _, batch in enumerate(pairs_batch_dev):
                pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
                pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)
       

                dev_loss = 0

                encoder_output, encoder_hidden, encoder_output_prob = encoder(pad_input_seqs, input_seq_lengths) 
                decoder_input = torch.ones(batch_size, 1).long().to(device)
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))


                teacher_forcing = True if random.random() <= tf_rate else False
                
                attn_weights = F.softmax(torch.ones(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)
 
                if teacher_forcing:
                    for i in range(0, pad_target_seqs.size(0)):
                        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, attn_weights)
                        target = pad_target_seqs.squeeze()
                        dev_loss += criterion(decoder_output, target[i])
                        decoder_input = pad_target_seqs[i].detach()
                else:
                    for i in range(0, pad_target_seqs.size(0)):     
                        decoder_output,  decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                        _, topi = decoder_output.topk(1)
                        target = pad_target_seqs.squeeze()
                        dev_loss += criterion(decoder_output, target[i])
                        decoder_input = topi.detach()

            
                # CTC LOSS
                targets = pad_target_seqs.squeeze().permute(1, 0)
                input_lengths = torch.ones(encoder_output_prob.size(1)) * encoder_output_prob.size(0)
                input_lengths = input_lengths.type(torch.LongTensor)
                target_seq_lengths = np.array(target_seq_lengths)
                target_lengths = torch.from_numpy(target_seq_lengths)
                
                dev_loss_ctc = ctc_loss(encoder_output_prob, targets, input_lengths, target_lengths)
                asr_loss = ((1 - ctc_weight) * dev_loss) + (ctc_weight * dev_loss_ctc)
                

                # NER BRANCH
                encoder_hidden_ner = (encoder_hidden[0][:2, :, :], encoder_hidden[1][:2, :, :])
                decoder_ner_output, decoder_ner_hidden = decoder_ner(pad_word_seqs, encoder_hidden_ner, word_seq_lengths, encoder_output)
                pad_tag_seqs = pad_tag_seqs.squeeze()
            
                mask = pad_tag_seqs.clone()
                mask[mask != 0] = 1
                mask = mask.byte()

                negative_log_likelihood = -decoder_ner.crf(decoder_ner_output, pad_tag_seqs, mask=mask)
            
                # interpolate the loss
                loss_dev = (0.8 * asr_loss) + (0.2 * negative_log_likelihood)
                
                batch_loss_dev += loss_dev.data

        
        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, (batch_loss_train.item() / (train_data_len / batch_size)), (batch_loss_dev.item() / (dev_data_len / batch_size))))

        with open('loss/english_ner_finetuned.txt', 'a') as f:
            f.write(str(epoch + 1) + '	' + str(batch_loss_train.item() / (train_data_len / batch_size)) + '  ' + str(batch_loss_dev.item() / (dev_data_len / batch_size)) + '\n')


        print('saving the models...')
        torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'decoder_ner': decoder_ner.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
        'decoder_ner_optimizer': decoder_ner_optimizer.state_dict(),
        }, 'weights/english_ner_finetuned/state_dict_' + str(epoch+1) + '.pt')


