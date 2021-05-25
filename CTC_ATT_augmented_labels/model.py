import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models

from torchcrf import CRF
import numpy as np



class Encoder(nn.Module):
    def __init__(self, input_tensor, hidden_size, num_layers, output_size, batch_size, device):
        super(Encoder, self).__init__()

        self.input_tensor = input_tensor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.output_size = output_size
        self.dropout = nn.Dropout(0.1)
        self.device = device
        
        self.out = nn.Linear(self.hidden_size, self.output_size)


        self.lstm = nn.LSTM(
                            self.input_tensor,
                            self.hidden_size,
                            num_layers=3,
                            bidirectional=True
                            )

        
        self.lstm_pyramid1 = nn.LSTM(
                             self.hidden_size,
                             self.hidden_size,
                             num_layers=1,
                             bidirectional=True
                            )
        

        self.lstm_pyramid2 = nn.LSTM(
                             self.hidden_size,
                             self.hidden_size,
                             num_layers=1,
                             bidirectional=True
                            )
        


    def forward(self, input_tensor, input_feature_lengths):
        input_tensor = pack_padded_sequence(input_tensor, input_feature_lengths)
         
        output, hidden = self.lstm(input_tensor) 
        output = pad_packed_sequence(output)[0]
        
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        
        output = self.reduce_resolution(output)    
        input_feature_lengths = np.floor_divide(np.array(input_feature_lengths), 2)
        output = pack_padded_sequence(output, input_feature_lengths)
        output, hidden = self.lstm_pyramid1(output)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:] 
      
        output = self.reduce_resolution(output)    
        input_feature_lengths = np.floor_divide(np.array(input_feature_lengths), 2)
        output = pack_padded_sequence(output, input_feature_lengths)
        output, hidden = self.lstm_pyramid2(output)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:] 
       
        output = self.dropout(output)
        output_prob = self.out(output)
        output_prob = output_prob.clamp(min=1e-6)
        output_prob = F.log_softmax(output, dim=2)
        #output_prob = []

        return output, hidden, output_prob


    def reduce_resolution(self, output):
        idx_odd = [j for j in range(output.size(0)) if j % 2 != 0]
        idx_even = [j for j in range(output.size(0)) if j % 2 == 0]
       
        output_odd = output[idx_odd, :, :]
        output_even = output[idx_even, :, :]
        if output_even.size(0) > output_odd.size(0):
            output_odd = torch.cat((output_odd, torch.zeros(1, output_odd.size(1), output_odd.size(2), device=self.device)))
        output = torch.mean(torch.stack([output_odd, output_even]), 0)

        return output



class EncoderSwedish(nn.Module):
    def __init__(self, input_tensor, hidden_size, num_layers, output_size, batch_size, device):
        super(EncoderSwedish, self).__init__()

        self.input_tensor = input_tensor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.output_size = output_size
        self.dropout = nn.Dropout(0.1)
        self.device = device
        
        self.out = nn.Linear(self.hidden_size, self.output_size)


        self.lstm = nn.LSTM(
                            self.input_tensor,
                            self.hidden_size,
                            num_layers=5,
                            bidirectional=True
                            )



    def forward(self, input_tensor, input_feature_lengths):
        input_tensor = pack_padded_sequence(input_tensor, input_feature_lengths)
         
        output, hidden = self.lstm(input_tensor) 
        output = pad_packed_sequence(output)[0]
        
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        
        output = self.dropout(output)
        output_prob = self.out(output)
        output_prob = output_prob.clamp(min=1e-6)
        output_prob = F.log_softmax(output, dim=2)

        return output, hidden, output_prob


    def reduce_resolution(self, output):
        idx_odd = [j for j in range(output.size(0)) if j % 2 != 0]
        idx_even = [j for j in range(output.size(0)) if j % 2 == 0]
       
        output_odd = output[idx_odd, :, :]
        output_even = output[idx_even, :, :]
        if output_even.size(0) > output_odd.size(0):
            output_odd = torch.cat((output_odd, torch.zeros(1, output_odd.size(1), output_odd.size(2), device=self.device)))
        output = torch.mean(torch.stack([output_odd, output_even]), 0)

        return output



    
# Attention decoder
class Decoder(nn.Module):
    def __init__(self, embedding_dim, encoder_hidden_size, attention_hidden_size, num_filters, output_size, num_layers, encoder_num_layers, batch_size, attention_type, device):
        super(Decoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.num_filters = num_filters
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.batch_size = batch_size
        self.attention_type = attention_type
        self.dropout = nn.Dropout(0.1)
        self.device = device

        self.embedding = nn.Embedding(output_size, embedding_dim)
         

        self.lstm = nn.LSTM(self.embedding_dim,
                            self.encoder_hidden_size,
                            num_layers=self.num_layers,
			    bidirectional=False)
        
        self.out = nn.Linear(self.encoder_hidden_size*2, self.output_size)    

        self.v = nn.Parameter(torch.FloatTensor(1, self.encoder_hidden_size).uniform_(-0.1, 0.1))
        self.b = nn.Parameter(torch.FloatTensor(self.encoder_hidden_size).uniform_(-0.1, 0.1))
        self.W_1 = torch.nn.Linear(self.encoder_hidden_size, self.attention_hidden_size)
        self.W_2 = torch.nn.Linear(self.encoder_hidden_size, self.attention_hidden_size)
        self.W_3 = nn.Linear(self.num_filters, self.attention_hidden_size)
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.num_filters, kernel_size=3, padding=1)
                
	
       
    def forward(self, input_tensor, decoder_hidden, encoder_output, attn_weights):
        embedding = self.embedding(input_tensor)
        embedding = embedding.permute(1, 0, 2)
       	           	
        decoder_output, decoder_hidden = self.lstm(embedding, decoder_hidden)
                        
        conv_feat = self.conv(attn_weights).permute(0, 2, 1) 
        conv_feat = conv_feat.permute(1, 0, 2)
        scores = self.hybrid_attention_score(encoder_output, decoder_output, conv_feat)

        scores = scores.permute(1, 0, 2)
        attn_weights = F.softmax(scores, dim=0)
        context = torch.bmm(attn_weights.permute(1, 2 ,0), encoder_output.permute(1, 0, 2))
        context = context.permute(1, 0, 2)
        output = torch.cat((context, decoder_output), -1)
        output = self.out(output[0])
 

        output = self.dropout(output)
        output = F.log_softmax(output, 1)
        return output, decoder_hidden, attn_weights.permute(1, 2, 0)


      
    def hybrid_attention_score(self, encoder_output, decoder_output, conv_feat):
        out = torch.tanh(self.W_1(decoder_output) + self.W_2(encoder_output) + self.W_3(conv_feat) + self.b)
        v = self.v.repeat(encoder_output.data.shape[1], 1).unsqueeze(1)
        out = out.permute(1, 0, 2)
        v = v.permute(0, 2, 1)
        scores = out.bmm(v)
        return scores

    
