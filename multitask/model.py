import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchcrf import CRF
import numpy as np



class Encoder(nn.Module):
    def __init__(self, input_tensor, hidden_size, num_layers, batch_size, device):
        super(Encoder, self).__init__()

        self.input_tensor = input_tensor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.1)
        self.device = device
        
        self.lstm = nn.LSTM(self.input_tensor,
                            self.hidden_size,
                            num_layers=1,
                            bidirectional=True
                            )

        self.lstm_pyramid = nn.LSTM(self.hidden_size,
                             self.hidden_size,
                             num_layers=1,
                             bidirectional=True
                            )



    def forward(self, input_tensor, input_feature_lengths):
        #pad the input timesteps to be divisible by 2
        if input_tensor.size(0) % 2 != 0:
            padding = torch.zeros(1, input_tensor.size(1), input_tensor.size(2), device=self.device)
            input_tensor = torch.cat((input_tensor, padding), dim=0)


        input_tensor = pack_padded_sequence(input_tensor, input_feature_lengths)
        output, hidden = self.lstm(input_tensor)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
        
        # pyramidal structure
        for i in range(1, self.num_layers):
            idx_odd = [j for j in range(output.size(0)) if j % 2 != 0]
            idx_even = [j for j in range(output.size(0)) if j % 2 == 0]

            output_odd = output[idx_odd, :, :]
            output_even = output[idx_even, :, :]
            if output_even.size(0) > output_odd.size(0):
                output_odd = torch.cat((output_odd, torch.zeros(1, output_odd.size(1), output_odd.size(2), device=self.device)))
            output = torch.mean(torch.stack([output_odd, output_even]), 0)
            
            input_feature_lengths = np.floor_divide(np.array(input_feature_lengths), 2)
            output = pack_padded_sequence(output, input_feature_lengths)
            output, hidden = self.lstm_pyramid(output)
            output = pad_packed_sequence(output)[0]
            output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
       
        output = self.dropout(output)
        
        return output, hidden





# Decoder with Luong attention
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, num_layers, encoder_num_layers, device):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.dropout = nn.Dropout(0.1)
        self.device = device

        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=False)

        self.out = nn.Linear(self.hidden_size*2, self.output_size)


    def forward(self, input_tensor, decoder_hidden, encoder_output):
        embedding = self.embedding(input_tensor)
        #embedding = self.dropout(embedding)
        embedding = embedding.permute(1, 0, 2)
    
        output, hidden = self.lstm(embedding, decoder_hidden)
 
        scores = torch.bmm(encoder_output.permute(1, 0, 2), output.permute(1, 2, 0))
        scores = scores.permute(1, 0, 2)
        attn_weights = F.softmax(scores, dim=0)

        context = torch.bmm(attn_weights.permute(1, 2 ,0), encoder_output.permute(1, 0, 2))
        context = context.permute(1, 0, 2)
        output = torch.cat((context, output), -1)
        output = self.out(output[0])
        output = self.dropout(output)
        output = F.log_softmax(output, 1)

        return output, hidden



class DecoderNER(nn.Module):
    def __init__(self, embedding_dim, hidden_size, tag_size, device):
        super(DecoderNER, self).__init__()

        self.tag_size = tag_size
        self.hidden_size = hidden_size
        self.device = device
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(0.1)    
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_size,
                            num_layers=1,
                            bidirectional=True
                            )
        self.out = nn.Linear(self.hidden_size, self.tag_size)
        self.crf = CRF(self.tag_size)


    def forward(self, input_tensor, decoder_hidden, word_seq_lengths):
        input_tensor = input_tensor.squeeze(-1)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(-1)
        
        #embedding = self.embedding(input_tensor)
        packed_embedding = pack_padded_sequence(input_tensor, word_seq_lengths, enforce_sorted=False)
        output, hidden = self.lstm(packed_embedding, decoder_hidden)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
        output = self.out(output)
        output = self.dropout(output)

        return output, hidden



