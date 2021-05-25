import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SOS_token = 1
EOS_token = 2
MAX_LENGTH = 400



class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6)
    
    def __lt__(self, other):
        return self.logp < other.logp


def beam_decode(decoder, target_tensor, decoder_hiddens, pad_target_seq_lengths, idx2char, attn_weights, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    # swedish width 8 topk 2
    beam_width = 8
    topk = 2 # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0), decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]]).to(device)

        decoder_context = torch.autograd.Variable(torch.zeros(1, 1, decoder.encoder_hidden_size*2)).to(device)

        # Number of sentences to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 5000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, attn_weights)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                #print(n.logp, log_p, (n.logp + log_p), n.leng+1, score)
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
            
            # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]
        
        utterances = []
        scores = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)
            
            utterance = utterance[::-1]
            utterances.append(utterance)
            scores.append(score)
        
        scores = scores[::-1]
        decoded_batch.append((utterances, scores))

    return decoded_batch
