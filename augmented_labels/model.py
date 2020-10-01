import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models

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
        

        self.lstm = nn.LSTM(
                            self.input_tensor,
                            self.hidden_size,
                            num_layers=1,
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


        self.lstm_pyramid3 = nn.LSTM(
                             self.hidden_size,
                             self.hidden_size,
                             num_layers=1,
                             bidirectional=True
                            )

        
        self.lstm_pyramid4 = nn.LSTM(
                             self.hidden_size,
                             self.hidden_size,
                             num_layers=1,
                             bidirectional=True
                            )



    def forward(self, input_tensor, input_feature_lengths):
        #pad the input timesteps to be divisible by 2
        #if input_tensor.size(0) % 2 != 0:
        #    padding = torch.zeros(1, input_tensor.size(1), input_tensor.size(2), device=self.device)
        #    input_tensor = torch.cat((input_tensor, padding), dim=0)

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
       

        output = self.reduce_resolution(output) 
        input_feature_lengths = np.floor_divide(np.array(input_feature_lengths), 2)
        output = pack_padded_sequence(output, input_feature_lengths)
        output, hidden = self.lstm_pyramid3(output)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:] 
        
        
        output = self.reduce_resolution(output) 
        input_feature_lengths = np.floor_divide(np.array(input_feature_lengths), 2)
        output = pack_padded_sequence(output, input_feature_lengths)
        output, hidden = self.lstm_pyramid4(output)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:] 
        
        
        output = self.dropout(output)
        
        return output, hidden


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
         

        if attention_type == 'additive':
            self.lstm = nn.LSTM(self.embedding_dim+self.encoder_hidden_size,
                                self.encoder_hidden_size,
                                num_layers=self.num_layers,
			        bidirectional=False)
            self.out = nn.Linear(self.encoder_hidden_size, self.output_size)
            
            self.v = nn.Parameter(torch.FloatTensor(1, self.encoder_hidden_size).uniform_(-0.1, 0.1))

            self.W_1 = torch.nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
            self.W_2 = torch.nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)


        else:
            self.lstm = nn.LSTM(self.embedding_dim,
                                self.encoder_hidden_size,
                                num_layers=self.num_layers,
			        bidirectional=False)
            self.out = nn.Linear(self.encoder_hidden_size*2, self.output_size)
            
            if attention_type == 'general':
                self.fc = nn.Linear(self.encoder_hidden_size, self.attention_hidden_size, bias=False)
            
            if attention_type == 'concat':
                self.fc = nn.Linear(self.encoder_hidden_size, self.attention_hidden_size, bias=False)
                self.v = nn.Parameter(torch.FloatTensor(1, self.encoder_hidden_size))

            if attention_type == 'hybrid':
                self.v = nn.Parameter(torch.FloatTensor(1, self.encoder_hidden_size).uniform_(-0.1, 0.1))
                self.b = nn.Parameter(torch.FloatTensor(self.encoder_hidden_size).uniform_(-0.1, 0.1))

                self.W_1 = torch.nn.Linear(self.encoder_hidden_size, self.attention_hidden_size, bias=False)
                self.W_2 = torch.nn.Linear(self.encoder_hidden_size, self.attention_hidden_size, bias=False)
                self.W_3 = nn.Linear(self.num_filters, self.attention_hidden_size, bias=False)
                self.conv = nn.Conv1d(in_channels=1, out_channels=self.num_filters, kernel_size=3, padding=1)
            
    
	
       
    def forward(self, input_tensor, decoder_hidden, encoder_output):
        embedding = self.embedding(input_tensor)
        embedding = embedding.permute(1, 0, 2)
       	
	# --- additive attention ---
        if self.attention_type == 'additive': 
            scores = self.hybrid_attention_score(encoder_output, decoder_hidden)
            attn_weights = F.softmax(scores, dim=0)
            context = torch.bmm(attn_weights.permute(0, 2, 1), encoder_output.permute(1, 0, 2))
            context = context.permute(1, 0, 2)
            output = torch.cat((context, embedding), -1)
            lstm_output, lstm_hidden = self.lstm(output, decoder_hidden)
            output = self.out(lstm_output.squeeze(0))
	# --- end additive attention ---	
 
           	
	# --- multiplicative attention ---
        else:
            decoder_output, decoder_hidden_new = self.lstm(embedding, decoder_hidden)
        
            if self.attention_type == 'dot':
                scores = self.dot_attention_score(encoder_output, decoder_output)

            if self.attention_type == 'general':
                scores = self.general_attention_score(encoder_output, decoder_output)
           
            if self.attention_type == 'concat':
                scores = self.concat_attention_score(encoder_output, decoder_output)

            if self.attention_type == 'hybrid':
                # add location-aware
                try:
                    conv_feat = self.conv(attn_weights).permute(0, 2, 1)
                except:
                    random_tensor = torch.rand(encoder_output.size(1), 1, encoder_output.size(0)).to(self.device)
                    conv_feat = self.conv(F.softmax(random_tensor, dim=-1)).to(self.device).permute(0, 2, 1)
 
                conv_feat = conv_feat.permute(1, 0, 2)
                scores = self.hybrid_attention_score(encoder_output, decoder_hidden[0], conv_feat)


            scores = scores.permute(1, 0, 2)
            attn_weights = F.softmax(scores, dim=0)
            context = torch.bmm(attn_weights.permute(1, 2 ,0), encoder_output.permute(1, 0, 2))
            context = context.permute(1, 0, 2)
            output = torch.cat((context, decoder_output), -1)
            output = self.out(output[0])
        # --- end multiplicative attention ---
 

        output = self.dropout(output)
        output = F.log_softmax(output, 1)

        return output, decoder_hidden_new


    def additive_attention_score(self, encoder_output, decoder_output):
        out = torch.tanh(self.W_1(decoder_output + encoder_output))
        v = self.v.repeat(encoder_output.data.shape[1], 1).unsqueeze(1)
        out = out.permute(1, 0, 2)
        v = v.permute(0, 2, 1)
        scores = out.bmm(v)
        return scores


    def dot_attention_score(self, encoder_output, decoder_output):
        scores = torch.bmm(encoder_output.permute(1, 0, 2), decoder_output.permute(1, 2, 0))
        return scores


    def general_attention_score(self, encoder_output, decoder_output):
        out = self.fc(decoder_output)
        encoder_output = encoder_output.permute(1, 0, 2)
        out = out.permute(1, 2, 0)
        scores = encoder_output.bmm(out)
        return scores

   
    def hybrid_attention_score(self, encoder_output, decoder_output, conv_feat):
        out = torch.tanh(self.W_1(decoder_output) + self.W_2(encoder_output) + self.W_3(conv_feat) + self.b)
        v = self.v.repeat(encoder_output.data.shape[1], 1).unsqueeze(1)
        out = out.permute(1, 0, 2)
        v = v.permute(0, 2, 1)
        scores = out.bmm(v)
        return scores

    
    def concat_attention_score(self, encoder_output, decoder_output):
        out = torch.tanh(self.fc(decoder_output + encoder_output))
        v = self.v.repeat(encoder_output.data.shape[1], 1).unsqueeze(1)
        out = out.permute(1, 0, 2)
        v = v.permute(0, 2, 1)
        scores = out.bmm(v)
        return scores





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


    def forward(self, input_tensor, decoder_hidden, word_seq_lengths, encoder_output):
        input_tensor = input_tensor.squeeze(-1)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(-1)
        
        packed_embedding = pack_padded_sequence(input_tensor, word_seq_lengths, enforce_sorted=False)
        output, hidden = self.lstm(packed_embedding, decoder_hidden)
        output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
        output = self.out(output)
        output = self.dropout(output)

        return output, hidden



