import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from onmt.modules import SRU

import math

class ParallelMyRNN(nn.Module):
  def __init__(self, input_size, hidden_size,
               num_layers=1, dropout=0, bidirectional=False):
    super(ParallelMyRNN, self).__init__()
    self.unit = SRU
    self.input_size = input_size
    self.rnn_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.Dropout = nn.Dropout(dropout)
    self.bidirectional = bidirectional
    self.num_directions = 2 if bidirectional else 1
    self.hidden_size = self.rnn_size * self.num_directions
    self.rnns = nn.ModuleList([nn.ModuleList() for _ in range(self.num_directions)])
    
    # for layer in range(num_layers):
    for layer in range(num_layers):
      layer_input_size = input_size if layer == 0 else self.hidden_size
      for direction in range(self.num_directions):
        self.rnns[direction].append(self.unit(layer_input_size, self.rnn_size, self.dropout))
  
  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.rnn_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)
  
  def initialize_parameters(self, param_init):
    for direction in range(self.num_directions):
      for layer in self.rnns[direction]:
        layer.initialize_parameters(param_init)
  
  def reverse_tensor(self, x, dim):
    idx = [i for i in range(x.size(dim) - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx))
    if x.is_cuda:
      idx = idx.cuda()
    return x.index_select(dim, idx)
  
  def forward(self, input, hidden=None):
    
    is_packed = isinstance(input, PackedSequence)
    if is_packed:
      input, batch_sizes = unpack(input)
      max_batch_size = batch_sizes[0]
    
    if hidden is None:
      # (num_layers x num_directions) x batch_size x rnn_size
      hidden = Variable(input.data.new(self.num_layers *
                                       self.num_directions,
                                       input.size(1),
                                       self.rnn_size).zero_(), requires_grad=False)
      if input.is_cuda:
        hidden = hidden.cuda()
    
    gru_out = []
    _input = input
    for i in range(self.num_layers):
      if not self.bidirectional:
        prev_layer = self.Dropout(_input)
        h = hidden[i]  # batch_size x rnn_size
        unit = self.rnns[0][i]  # Computation unit
        
        layer_out, hid_uni = unit(prev_layer, h)  # src_len x batch x hidden_size
      
      else:
        input_forward = self.Dropout(_input)
        input_backward = self.Dropout(_input)
        h_forward = hidden[i * self.num_directions]  # batch_size x rnn_size
        h_backward = hidden[i * self.num_directions + 1]  # batch_size x rnn_size
        unit_forward = self.rnns[0][i]  # Computation unit
        unit_backward = self.rnns[1][i]  # Computation unit
        
        output_forward, h_forward = unit_forward(input_forward, h_forward)
        output_backward, h_backward = self.compute_backwards(unit_backward, input_backward, h_backward)
        
        layer_out = torch.cat([output_forward, output_backward], dim=2)  # src_len x batch x hidden_size
      
      _input = layer_out
      
      if self.bidirectional:
        gru_out.append(output_forward[-1].unsqueeze(0))
        gru_out.append(output_backward[-1].unsqueeze(0))  # num_directions x [batch x rnn_size]
      else:
        gru_out.append(layer_out)
    
    hidden = torch.cat(gru_out, dim=0)  # (num_layers x num_directions) x batch x rnn_size
    
    output = _input
    
    return output, hidden
  
  def __repr__(self):
    s = '{name}({input_size}, {rnn_size}'
    if self.num_layers != 1:
      s += ', num_layers={num_layers}'
    if self.dropout != 0:
      s += ', dropout={dropout}'
    if self.bidirectional is not False:
      s += ', bidirectional={bidirectional}'
    s += ')'
    return s.format(name=self.__class__.__name__, **self.__dict__)
  
  def compute_backwards(self, unit, input, hidden):
    h = hidden
    steps = torch.cat(input.split(1, dim=0)[::-1], dim=0)
    out, hidden = unit(steps, h)
    out = torch.cat(out.split(1, dim=0)[::-1], dim=0)
    return out, hidden
