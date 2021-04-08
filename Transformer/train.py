import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.early_stopping import EarlyStopping

import gc

def train(pairs_batch_train, pairs_batch_dev, transformer, criterion, optimizer, num_epochs, batch_size, train_data_len, dev_data_len, device):
    accumulate_steps = 6

    for epoch in range(100):
        transformer.train()
        
        batch_loss_train = 0 
        batch_loss_dev = 0
    
        #for iteration, batch in enumerate(pairs_batch_train):
        #    train_loss = 0
        #    optimizer.zero_grad()

        #    pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
        #    pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)
        #    
        #    output = transformer(pad_input_seqs, input_seq_lengths, pad_target_seqs)

        #    output = output.permute(1, 2, 0)
        #    targets = pad_target_seqs.permute(1, 0, 2).squeeze()

        #    train_loss = criterion(output[:, :, :-1], targets[:, 1:])
        #    batch_loss_train += train_loss.item()
        #    
        #    train_loss.backward()
        #    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)
        #    optimizer.step()
        

        # gradient accumulation
        optimizer.zero_grad()
        for iteration, batch in enumerate(pairs_batch_train):
            train_loss = 0
     
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
            pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)
            
            output = transformer(pad_input_seqs, input_seq_lengths, pad_target_seqs)

            output = output.permute(1, 2, 0)
            targets = pad_target_seqs.permute(1, 0, 2).squeeze()

            train_loss = criterion(output[:, :, :-1], targets[:, 1:])
            train_loss = train_loss / accumulate_steps
            train_loss.backward()
            batch_loss_train += train_loss.detach().item()

            if (iteration + 1) % accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()


        
        # VALIDATION
        with torch.no_grad():
            transformer.eval()

            for iteration, batch in enumerate(pairs_batch_dev):
                dev_loss = 0

                pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_word_seqs, word_seq_lengths, pad_tag_seqs, tag_seq_lengths = batch
                pad_input_seqs, pad_target_seqs, pad_word_seqs, pad_tag_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_word_seqs.to(device), pad_tag_seqs.to(device)
                
                
                output = transformer(pad_input_seqs, input_seq_lengths, pad_target_seqs)
   
                output = output.permute(1, 2, 0)
                targets = pad_target_seqs.permute(1, 0, 2).squeeze()
            
                dev_loss = criterion(output[:, :, :-1], targets[:, 1:])
   
                batch_loss_dev += dev_loss.item()
               

        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, (batch_loss_train / len(pairs_batch_train)), (batch_loss_dev / len(pairs_batch_dev))))
        
        #with open('loss/swedish.txt', 'a') as f:
        #    f.write(str(epoch + 1) + '  ' + str(batch_loss_train / len(pairs_batch_train)) + '  ' + str(batch_loss_dev / len(pairs_batch_dev)) + '\n')

        #
        #print('saving the model...')
        #torch.save({
        #'transformer': transformer.state_dict(),
        #'optimizer': optimizer.state_dict(),
        #}, 'weights/swedish/state_dict_' + str(epoch+1) + '.pt')



