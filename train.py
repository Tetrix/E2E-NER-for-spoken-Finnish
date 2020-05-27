import random
import torch
from utils.early_stopping import EarlyStopping



def train(pairs_batch_train, pairs_batch_dev, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, num_epochs, device):
    clip = 50.0
    tf_rate = 1
    early_stopping = EarlyStopping(patience=15, verbose=False, delta=0)

    for epoch in range(56, 70):   
        encoder.train()
        decoder.train()

        for _, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, pad_target_seqs_lengths = batch
            pad_input_seqs, pad_target_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device)
            
            train_loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
        
            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)

            decoder_input = torch.ones(batch_size, 1).long().to(device) 
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
            #decoder_hidden = encoder_hidden[0].sum(0, keepdim=True)

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
                    

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            encoder_optimizer.step()
            decoder_optimizer.step()



        # CALCULATE EVALUATION
        with torch.no_grad():
            for _, batch in enumerate(pairs_batch_dev):
                encoder.eval()
                decoder.eval()

                pad_input_seqs, input_seq_lengths, pad_target_seqs, pad_target_seqs_lengths = batch
                pad_input_seqs, pad_target_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device)

                dev_loss = 0

                encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
            
                decoder_input = torch.ones(batch_size, 1).long().to(device)
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
                #decoder_hidden = encoder_hidden[0].sum(keepdim=True)

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




        #early_stopping(complete_loss_dev, (encoder, decoder, encoder_optimizer, decoder_optimizer))
        #if early_stopping.early_stop:
        #    print('Early stopping')
        #    break

        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, train_loss.item(), dev_loss.item()))

        with open('loss_new.txt', 'a') as f:
            f.write(str(train_loss.item()) + '  ' + str(dev_loss.item()) + '\n')


        print('saving the models...')
        torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict()
        }, 'weights/new/state_dict_' + str(epoch+1) + '.pt')
        #}, 'weights/state_dict.pt')

