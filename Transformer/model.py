import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from torchcrf import CRF
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x = x + self.scale * self.pe[:x.size(0), :]
        x += self.pe[:x.size(0), :]

        return self.dropout(x)



class Encoder(nn.Module):
    def __init__(self, input_size, n_head_encoder, d_model, n_layers_encoder, max_len):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.n_head_encoder = n_head_encoder
        self.d_model = d_model
        self.n_layers_encoder = n_layers_encoder
        self.max_len = max_len

        # define the layers
        self.lin_transform = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_len)
        encoder_layers = TransformerEncoderLayer(self.d_model, self.n_head_encoder, self.d_model*4, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.n_layers_encoder)


    def forward(self, input_tensor, mask, src_key_padding_mask):
        input_tensor = self.lin_transform(input_tensor)
        input_tensor *= math.sqrt(self.d_model)
        input_encoded = self.pos_encoder(input_tensor)
        output = self.transformer_encoder(input_encoded, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output



class Decoder(nn.Module):
    def __init__(self, n_tokens, n_head_decoder, d_model, n_layers_decoder, max_len):
        super(Decoder, self).__init__()
        
        self.n_tokens = n_tokens
        self.n_head_decoder = n_head_decoder
        self.d_model = d_model
        self.n_layers_decoder = n_layers_decoder
        self.max_len = max_len

        self.embedding = nn.Embedding(self.n_tokens, self.d_model)
        self.pos_decoder = PositionalEncoding(self.d_model, self.max_len)
        decoder_layers = TransformerDecoderLayer(self.d_model, self.n_head_decoder, self.d_model*4, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, self.n_layers_decoder)
        

    def forward(self, tgt, memory, memory_mask, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask):
        #tgt = tgt[4:5, 1, :]
        embedded = self.embedding(tgt)
        try:
            embedded = embedded.squeeze(2)
        except:
            pass
        
        embedded *= math.sqrt(self.d_model)
        target_encoded = self.pos_decoder(embedded)
        
        output = self.transformer_decoder(
            tgt=target_encoded,
            memory=memory, 
            tgt_mask=tgt_mask, 
            memory_key_padding_mask=memory_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask)

        return output



class Transformer(nn.Module):
    def __init__(self, input_size, n_tokens, n_head_encoder, n_head_decoder, d_model, n_layers_encoder, n_layers_decoder, max_len):
        super(Transformer, self).__init__()
         
        self.encoder = Encoder(input_size, n_head_encoder, d_model, n_layers_encoder, max_len)
        self.decoder = Decoder(n_tokens, n_head_decoder, d_model, n_layers_decoder, max_len)
        self.out = nn.Linear(d_model, n_tokens)

    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
    def generate_subsequent_mask(self, seq):
        len_s = seq.size(0)
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask


    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        src_mask = self.generate_square_subsequent_mask(src_seq_len).to(device=src.device)

        src_padding_mask = (src == 0).transpose(0, 1)
        tgt_padding_mask = (tgt == 0).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask[:, :, 0], tgt_padding_mask.squeeze(-1)


    def forward(self, input_seq, input_seq_lengths, target_seq):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(input_seq, target_seq)
        
        memory = self.encoder(input_seq, mask=src_mask, src_key_padding_mask=src_padding_mask)

        decoder_output = self.decoder(
                                        tgt=target_seq, 
                                        memory=memory, 
                                        memory_mask=None,
                                        tgt_mask=tgt_mask,
                                        memory_key_padding_mask=src_padding_mask,
                                        tgt_key_padding_mask=tgt_padding_mask
                                        )

        output = self.out(decoder_output)
        return output




