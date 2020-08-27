import random
import torch
from utils.early_stopping import EarlyStopping



def train(pairs_batch_train, pairs_batch_dev, encoder, decoder, decoder_ner, encoder_optimizer, decoder_optimizer, decoder_ner_optimizer, criterion, batch_size, num_epochs, device):
    clip = 50.0
    tf_rate = 1
    lambda_factor = 0.80
    early_stopping = EarlyStopping(patience=10, verbose=False, delta=0)

    for epoch in range(50):
        encoder.train()
        decoder.train()
        decoder_ner.train()
 
        for iteration, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
            pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)
            
            train_loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            decoder_ner_optimizer.zero_grad()
           
            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
            decoder_input = torch.ones(batch_size, 1).long().to(device)
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))

            teacher_forcing = True if random.random() <= tf_rate else False
            
            if teacher_forcing:
                for i in range(0, pad_target_seqs.size(0)):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                    target = pad_target_seqs.squeeze()
                    train_loss += criterion(decoder_output, target[i])
                    decoder_input = pad_target_seqs[i]
            else:
                for i in range(0, pad_target_seqs.size(0)):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                    _, topi = decoder_output.topk(1)
                    target = pad_target_seqs.squeeze()
                    train_loss += criterion(decoder_output, target[i])
                    decoder_input = topi.detach()


                        
            # NER BRANCH
            encoder_hidden_ner = (encoder_hidden[0][:2, :, :], encoder_hidden[1][:2, :, :])
            decoder_ner_output, decoder_ner_hidden = decoder_ner(pad_word_seqs, encoder_hidden_ner, word_seq_lengths, encoder_output)
            pad_tag_seqs = pad_tag_seqs.squeeze()
            
            mask = pad_tag_seqs.clone()
            mask[mask != 0] = 1
            mask = mask.byte()

            negative_log_likelihood = -decoder_ner.crf(decoder_ner_output, pad_tag_seqs, mask=mask)
            
            loss = (lambda_factor * train_loss) +  ((1 - lambda_factor) * negative_log_likelihood)
            #loss = negative_log_likelihood
            

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

                encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths) 
                decoder_input = torch.ones(batch_size, 1).long().to(device)
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))


                teacher_forcing = True if random.random() <= tf_rate else False
            
                if teacher_forcing:
                    for i in range(0, pad_target_seqs.size(0)):
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                        target = pad_target_seqs.squeeze()
                        dev_loss += criterion(decoder_output, target[i])
                        decoder_input = pad_target_seqs[i]
                else:
                    for i in range(0, pad_target_seqs.size(0)):     
                        decoder_output,  decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                        _, topi = decoder_output.topk(1)
                        target = pad_target_seqs.squeeze()
                        dev_loss += criterion(decoder_output, target[i])
                        decoder_input = topi.detach()


            
            
                # NER BRANCH
                encoder_hidden_ner = (encoder_hidden[0][:2, :, :], encoder_hidden[1][:2, :, :])
                decoder_ner_output, decoder_ner_hidden = decoder_ner(pad_word_seqs, encoder_hidden_ner, word_seq_lengths, encoder_output)
                pad_tag_seqs = pad_tag_seqs.squeeze()

                mask = pad_tag_seqs.clone()
                mask[mask != 0] = 1
                mask = mask.byte()

                negative_log_likelihood_dev = -decoder_ner.crf(decoder_ner_output, pad_tag_seqs, mask=mask)
                loss_dev = (lambda_factor * dev_loss) +  ((1 - lambda_factor) * negative_log_likelihood_dev)
                #loss_dev = negative_log_likelihood_dev


        #early_stopping(complete_loss_dev, (encoder, decoder, encoder_optimizer, decoder_optimizer))
        #if early_stopping.early_stop:
        #    print('Early stopping')
        #    break

        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, loss.item(), loss_dev.item()))


        with open('loss/loss_whole_normalized_ner.txt', 'a') as f:
            f.write(str(epoch + 1) + '	' + str(loss.item()) + '  ' + str(loss_dev.item()) + '\n')


        print('saving the models...')
        torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'decoder_ner': decoder_ner.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
        'decoder_ner_optimizer': decoder_ner_optimizer.state_dict()
        }, 'weights/whole_normalized_ner/state_dict_' + str(epoch+1) + '.pt')
        #}, 'weights/state_dict.pt')


