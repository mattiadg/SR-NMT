
import torch
import torch.nn as nn

import onmt
from .modules.SRU_units import BiSRU

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from onmt.modules.Units import ParallelMyRNN

def getEncoder(encoder_type):
    encoders = {'RNN': Encoder,
                'SGU': SGUEncoder}
    if encoder_type not in encoders:
        raise NotImplementedError(encoder_type)
    return encoders[encoder_type]


class Encoder(nn.Module):

    def __init__(self, opt, dicts):

        def getunittype(rnn_type):
            if rnn_type in ['LSTM', 'GRU']:
                return getattr(nn, rnn_type)
            elif rnn_type == 'SRU':
                return ParallelMyRNN

        self.layers = opt.layers_enc
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)

        rnn_type = opt.rnn_encoder_type if opt.rnn_encoder_type else opt.rnn_type
        self.rnn = getunittype(rnn_type)(
                opt.word_vec_size, self.hidden_size,
                num_layers=opt.layers_enc, dropout=opt.dropout,
                bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def initialize_parameters(self, param_init):
        if hasattr(self.rnn, 'initialize_parameters'):
            self.rnn.initialize_parameters(param_init)

    def forward(self, input, hidden=None):

        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            emb = pack(self.word_lut(input[0]), lengths)
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(outputs, PackedSequence):
            outputs = unpack(outputs)[0]

        return hidden_t, outputs, emb

class StackedSGU(nn.Module):

    def __init__(self, layers, input_size, hidden_size, layer_norm, dropout):
        self.layers = layers
        super(StackedSGU, self).__init__()
        self.sgus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(layers):
            self.sgus.append(BiSRU(input_size, hidden_size, layer_norm, dropout))
            input_size = hidden_size

    def initialize_parameters(self, param_init):
        for sgu in self.sgus:
            sgu.initialize_parameters(param_init)

    def forward(self, input):

        hiddens = []
        for i in range(self.layers):
            input = self.sgus[i](input)
            hiddens += [input[-1]]
        return input, torch.stack(hiddens)

class SGUEncoder(nn.Module):

    def __init__(self, opt, dicts):

        self.layers = opt.layers_enc
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions

        super(SGUEncoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.sgu = StackedSGU(self.layers, opt.word_vec_size,
                              self.hidden_size * self.num_directions, opt.layer_norm,
                              opt.dropout)


    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def initialize_parameters(self, param_init):
        self.sgu.initialize_parameters(param_init)

    def forward(self, input, hidden=None):

        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            emb = self.word_lut(input[0])
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.sgu(emb)

        return hidden_t, outputs, emb
