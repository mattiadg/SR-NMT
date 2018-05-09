import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import onmt

from onmt.modules import SRU

from .modules.SRU_units import AttSRU
from .modules.Attention import getAttention
from .modules.Normalization import LayerNorm
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def getDecoder(decoderType):
    decoders = {'StackedRNN': StackedRNNDecoder,
                'SGU': SGUDecoder,
                'ParallelRNN': ParallelRNNDecoder}

    if decoderType not in decoders:
        raise NotImplementedError(decoderType)

    return decoders[decoderType]


def getStackedLayer(rnn_type):
    if rnn_type == "LSTM":
        return StackedLSTM
    elif rnn_type == "GRU":
        return StackedGRU
    else:
        return None

def getRNN(rnn_type):
    rnns = {'LSTM': nn.LSTM,
            'GRU': nn.GRU
            }

    return rnns[rnn_type]

class StackedLSTM(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class StackedSGU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, layer_norm, dropout):
        super(StackedSGU, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            self.layers.append(AttSRU(input_size,
                                    rnn_size, rnn_size, layer_norm, dropout))
            input_size = rnn_size

    def initialize_parameters(self, param_init):
        for layer in self.layers:
            layer.initialize_parameters(param_init)

    def forward(self, dec_state, hidden, enc_out):
        input = dec_state
        first_input = dec_state
        hiddens = []
        for i, layer in enumerate(self.layers):
            input, new_hidden, attn_state = layer(input, hidden[i], enc_out)
            hiddens += [new_hidden]

        return self.dropout(input), torch.stack(hiddens), attn_state


class StackedRNNDecoder(nn.Module):

    def __init__(self, opt, dicts):

        self.layers = opt.layers_dec
        self.input_feed = opt.input_feed
        self.hidden_size = opt.rnn_size

        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(StackedRNNDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)

        rnn_type = opt.rnn_decoder_type if opt.rnn_decoder_type else opt.rnn_type
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getStackedLayer(rnn_type)\
                (opt.layers_dec, input_size, opt.rnn_size, opt.dropout)
        else:
            self.rnn = getStackedLayer(rnn_type) \
                (opt.layers_dec, input_size, opt.rnn_size, opt.activ,
                 opt.layer_norm, opt.dropout)

        self.attn = getAttention(opt.attn_type)(opt.rnn_size, opt.activ)

        self.linear_ctx = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.linear_out = nn.Linear(2 * opt.rnn_size, opt.rnn_size)

        self.dropout = nn.Dropout(opt.dropout)
        self.log = self.rnn.log if hasattr(self.rnn, 'log') else False

        self.layer_norm = opt.layer_norm
        if self.layer_norm:
            self.ctx_ln = LayerNorm(opt.rnn_size)

        self.activ = getattr(F, opt.activ)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def initialize_parameters(self, param_init):
        pass

    def forward(self, input, hidden, context, init_output):
        """
        input: targetL x batch
        hidden: batch x hidden_dim
        context: sourceL x batch x hidden_dim
        init_output: batch x hidden_dim
        """
        # targetL x batch x hidden_dim
        emb = self.word_lut(input)

        # batch x sourceL x hidden_dim
        context = context.transpose(0, 1)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output

        for emb_t in emb.split(1):
            # batch x word_dim
            emb_inp = emb_t.squeeze(0)

            if self.input_feed == 1:
                # batch x (word_dim+hidden_dim)
                emb_inp_feed = torch.cat([emb_inp, output], 1)
            else:
                emb_inp_feed = emb_inp

            # batch x hidden_dim, layers x batch x hidden_dim
            if self.log:
                rnn_output, hidden, activ = self.rnn(emb_inp_feed, hidden)
            else:
                rnn_output, hidden = self.rnn(emb_inp_feed, hidden)

            values = context
            pctx = self.linear_ctx(self.dropout(context))
            if self.layer_norm:
                pctx = self.ctx_ln(pctx)
            weightedContext, attn = self.attn(rnn_output, pctx, values)

            contextCombined = self.linear_out(torch.cat([rnn_output, weightedContext], dim=-1))

            output = self.activ(contextCombined)
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)

        if self.log:
            return outputs, hidden, attn, activ

        return outputs, hidden, attn


class SGUDecoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers_dec
        self.hidden_size = opt.rnn_size

        input_size = opt.word_vec_size

        super(SGUDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)

        #self.rnn = getRNN(opt.rnn_type)(opt.word_vec_size, opt.rnn_size, num_layers=1,
        #                  dropout=opt.dropout)
        self.linear = nn.Linear(opt.word_vec_size, self.hidden_size)
        self.gate = nn.Linear(self.hidden_size, self.hidden_size)

        self.stacked = StackedSGU(opt.layers_dec, opt.rnn_size,
                                  opt.rnn_size, opt.layer_norm,
                                  opt.dropout)

        self.log = False

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def initialize_parameters(self, param_init):
        self.stacked.initialize_parameters(param_init)
        #self.attn.initialize_parameters(param_init)

    def forward(self, input, hidden, context, init_output):
        """
        input: targetL x batch
        hidden: num_layers x batch x hidden_dim
        context: sourceL x batch x hidden_dim
        init_output: batch x hidden_dim
        """
        batch_size = input.size(1)
        hidden_dim = context.size(2)

        #targetL x batch x hidden_dim
        emb = self.word_lut(input)

        # batch x sourceL x hidden_dim
        context = context.transpose(0, 1)
        if len(hidden.size()) < 3:
            hidden = hidden.unsqueeze(0)

        # (targetL x batch) x sourceL x hidden_dim
        #values = context.repeat(emb.size(0), 1, 1)
        rnn_outputs = emb #.view(-1, hidden_dim)

        outputs, hidden, attn = self.stacked(rnn_outputs, hidden, context)

        return outputs, hidden, attn


class StackedSRU(nn.Module):
  def __init__(self, num_layers, input_size, rnn_size, dropout):
    super(StackedSRU, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.num_layers = num_layers
    self.layers = nn.ModuleList()
    
    for i in range(num_layers):
      self.layers.append(SRU(input_size, rnn_size, dropout))
      input_size = rnn_size
  
  def initialize_parameters(self, param_init):
    for layer in self.layers:
      layer.initialize_parameters(param_init)
  
  def forward(self, input, hidden):
    """

    :param input: batch x hi
    :param hidden:
    :return:
    """
    h_1 = []
    for i, layer in enumerate(self.layers):
      h_1_i, h = layer(input, hidden[i])
      input = h_1_i
      h_1 += [h]
    
    h_1 = torch.stack(h_1)
    
    return input, h_1


class ParallelRNNDecoder(nn.Module):
  def __init__(self, opt, dicts):
    from .modules.Attention import MLPAttention
    
    self.layers = opt.layers_dec
    self.hidden_size = opt.rnn_size
    
    input_size = opt.word_vec_size
    
    super(ParallelRNNDecoder, self).__init__()
    self.word_lut = nn.Embedding(dicts.size(),
                                 opt.word_vec_size,
                                 padding_idx=onmt.Constants.PAD)
    
    self.rnn = StackedSRU(self.layers, input_size, self.hidden_size, opt.dropout)
    
    self.attn = MLPAttention(opt.rnn_size, opt.activ)  # getAttention(opt.attn_type)(opt.rnn_size, opt.activ)
    
    self.linear_ctx = nn.Linear(opt.rnn_size, opt.rnn_size)
    self.linear_out = nn.Linear(2 * opt.rnn_size, opt.rnn_size)
    
    self.dropout = nn.Dropout(opt.dropout)
  
  def load_pretrained_vectors(self, opt):
    if opt.pre_word_vecs_dec is not None:
      pretrained = torch.load(opt.pre_word_vecs_dec)
      self.word_lut.weight.data.copy_(pretrained)
  
  def initialize_parameters(self, param_init):
    pass
  
  def forward(self, input, hidden, context, init_output):
    """
    input: targetL x batch
    hidden: batch x hidden_dim
    context: sourceL x batch x hidden_dim
    init_output: batch x hidden_dim
    """
    # targetL x batch x hidden_dim
    emb = self.word_lut(input)
    
    # batch x sourceL x hidden_dim
    context = context.transpose(0, 1)
    
    # batch x hidden_dim, layers x batch x hidden_dim
    rnn_output, hidden = self.rnn(emb, hidden)
    
    values = context
    pctx = self.linear_ctx(self.dropout(context))
    
    weightedContext, attn = self.attn(self.dropout(rnn_output), pctx, values)
    
    contextCombined = self.linear_out(self.dropout(torch.cat([rnn_output, weightedContext], dim=-1)))
    
    output = F.tanh(contextCombined)
    output = self.dropout(output)
    
    return output, hidden, attn
