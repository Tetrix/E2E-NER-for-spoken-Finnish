import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models

from torchcrf import CRF
import numpy as np


class VggExtractor(nn.Module):
    def __init__(self, vgg16, device):
        super(VggExtractor, self).__init__()

        self.vgg16_model = vgg16
        self.device = device
        self.lin = nn.Linear(512, 128)


    def forward(self, input_tensor):
        input_tensor = input_tensor.permute(1, 0, 2).unsqueeze(1)
        vgg_output = self.vgg16_model.features(input_tensor)
        vgg_output = vgg_output.squeeze(-1).permute(0, 2, 1)
        vgg_output = F.relu(self.lin(vgg_output))
        input_tensor = vgg_output.squeeze(-1).permute(1, 0, 2)
        
        return input_tensor




class Encoder(nn.Module):
    def __init__(self, input_tensor, hidden_size, num_layers, batch_size, device):
        super(Encoder, self).__init__()

        self.input_tensor = input_tensor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.1)
        self.device = device
        
        #self.input_tensor = 128

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
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

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




# Attention decoder
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, num_layers, encoder_num_layers, batch_size, attention_type, device):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
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
            self.lstm = nn.LSTM(self.embedding_dim+self.hidden_size,
                                self.hidden_size,
                                num_layers=self.num_layers,
			        bidirectional=False)
            self.out = nn.Linear(self.hidden_size, self.output_size)
            
            self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size).uniform_(-0.1, 0.1))

            self.W_1 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.W_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)


        else:
            self.lstm = nn.LSTM(self.embedding_dim,
                                self.hidden_size,
                                num_layers=self.num_layers,
			        bidirectional=False)
            self.out = nn.Linear(self.hidden_size*2, self.output_size)
            
            if attention_type == 'general':
                self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            
            if attention_type == 'concat':
                self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

            if attention_type == 'hybrid':
                self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size).uniform_(-0.1, 0.1))
                self.b = nn.Parameter(torch.FloatTensor(self.hidden_size).uniform_(-0.1, 0.1))

                self.W_1 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.W_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.W_3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.conv = nn.Conv1d(in_channels=1, out_channels=self.hidden_size, kernel_size=3, padding=1)
            
    
	
       
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
            lstm_output, lstm_hidden = self.lstm(embedding, decoder_hidden)
        
            if self.attention_type == 'dot':
                scores = self.dot_attention_score(encoder_output, lstm_output)

            if self.attention_type == 'general':
                scores = self.general_attention_score(encoder_output, decoder_hidden)
           
            if self.attention_type == 'concat':
                scores = self.concat_attention_score(encoder_output, decoder_hidden)

            if self.attention_type == 'hybrid':
                # add location-aware
                try:
                    conv_feat = self.conv(attn_weights).permute(0, 2, 1)
                except:
                    conv_feat = self.conv(torch.rand(encoder_output.size(1), 1, encoder_output.size(0)).to(self.device)).permute(0, 2, 1)
   
                conv_feat = conv_feat.permute(1, 0, 2)
                scores = self.hybrid_attention_score(encoder_output, decoder_hidden, conv_feat)
 

            scores = scores.permute(1, 0, 2)
            attn_weights = F.softmax(scores, dim=0)
            context = torch.bmm(attn_weights.permute(1, 2 ,0), encoder_output.permute(1, 0, 2))
            context = context.permute(1, 0, 2)
            output = torch.cat((context, lstm_output), -1)
            output = self.out(output[0])
        # --- end multiplicative attention ---
 

        output = self.dropout(output)
        output = F.log_softmax(output, 1)

        return output, lstm_hidden


    def additive_attention_score(self, encoder_output, decoder_hidden):
        out = torch.tanh(self.W_1(decoder_hidden[0] + encoder_output))
        v = self.v.repeat(encoder_output.data.shape[1], 1).unsqueeze(1)
        out = out.permute(1, 0, 2)
        v = v.permute(0, 2, 1)
        scores = out.bmm(v)
        return scores


    def dot_attention_score(self, encoder_output, lstm_output):
        scores = torch.bmm(encoder_output.permute(1, 0, 2), lstm_output.permute(1, 2, 0))
        return scores


    def general_attention_score(self, encoder_output, decoder_hidden):
        out = self.fc(decoder_hidden[0])
        encoder_output = encoder_output.permute(1, 0, 2)
        out = out.permute(1, 2, 0)
        scores = encoder_output.bmm(out)
        return scores

    
    def hybrid_attention_score(self, encoder_output, decoder_hidden, conv_feat):
        out = torch.tanh(self.W_1(decoder_hidden[0]) + self.W_2(encoder_output) + self.W_3(conv_feat) + self.b)
        v = self.v.repeat(encoder_output.data.shape[1], 1).unsqueeze(1)
        out = out.permute(1, 0, 2)
        v = v.permute(0, 2, 1)
        scores = out.bmm(v)
        return scores

    
    def concat_attention_score(self, encoder_output, decoder_hidden):
        out = torch.tanh(self.fc(decoder_hidden[0] + encoder_output))
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



