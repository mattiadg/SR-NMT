"""
Context gate is a decoder module that takes as input the previous word
embedding, the current decoder state and the attention state, and produces a
gate.
The gate can be used to select the input from the target side context
(decoder state), from the source context (attention state) or both.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Normalization import LayerNorm
from torch.autograd import Variable
import numpy as np


class AttSRU(nn.Module):

    def __init__(self, input_size, attention_size, output_size, layer_norm, dropout):
        from .Attention import MLPAttention
        super(AttSRU, self).__init__()
        self.linear_in = nn.Linear(input_size, 3*output_size, bias=(not layer_norm))
        self.linear_hidden = nn.Linear(output_size, output_size, bias=(not layer_norm))
        self.linear_ctx = nn.Linear(output_size, output_size, bias=(not layer_norm))
        self.linear_enc = nn.Linear(output_size, output_size, bias=(not layer_norm))
        self.output_size = output_size
        self.attn = MLPAttention(attention_size, layer_norm=True)
        self.layer_norm = layer_norm
        self.dropout = nn.Dropout(dropout)
        if self.layer_norm:
            self.preact_ln = LayerNorm(3 * output_size)
            self.enc_ln = LayerNorm(output_size)

            self.trans_h_ln = LayerNorm(output_size)
            self.trans_c_ln = LayerNorm(output_size)

    def initialize_parameters(self, param_init):
        self.preact_ln.initialize_parameters(param_init)
        self.trans_h_ln.initialize_parameters(param_init)
        self.trans_c_ln.initialize_parameters(param_init)
        self.enc_ln.initialize_parameters(param_init)

    def forward(self, prev_layer, hidden, enc_output):
        """
        :param prev_layer: targetL x batch x output_size
        :param hidden: batch x output_size
        :param enc_output: (targetL x batch) x sourceL x output_size
        :return:
        """

        # targetL x batch x output_size
        preact = self.linear_in(self.dropout(prev_layer))
        pctx = self.linear_enc(self.dropout(enc_output))
        if self.layer_norm:
            preact = self.preact_ln(preact)
            pctx = self.enc_ln(pctx)
            #z = self.z_ln(z)
            #prev_layer_t = self.prev_layer_ln(prev_layer_t)
            #h_gate = self.h_gate_ln(h_gate)
        z, h_gate, prev_layer_t = preact.split(self.output_size, dim=-1)
        z, h_gate = F.sigmoid(z), F.sigmoid(h_gate)

        ss = []
        for i in range(prev_layer.size(0)):
            s = (1. - z[i]) * hidden + z[i] * prev_layer_t[i]
            # targetL x batch x output_size
            ss += [s]
            # batch x output_size
            hidden = s

        # (targetL x batch) x output_size
        ss = torch.stack(ss)
        attn_out, attn = self.attn(self.dropout(ss), pctx, pctx)
        attn_out = attn_out / np.sqrt(self.output_size)

        trans_h = self.linear_hidden(self.dropout(ss))
        trans_c = self.linear_ctx(self.dropout(attn_out))
        if self.layer_norm:
            #out = self.post_ln(out)
            trans_h = self.trans_h_ln(trans_h)
            trans_c = self.trans_c_ln(trans_c)
        #trans_h, trans_c = F.tanh(trans_h), F.tanh(trans_c)
        out = trans_h + trans_c
        out = F.tanh(out)
        out = out.view(prev_layer.size())
        out = (1. - h_gate) * out + h_gate * prev_layer

        return out, hidden, attn

class BiSRU(nn.Module):

    def __init__(self, input_size, output_size, layer_norm, dropout):
        super(BiSRU, self).__init__()
        self.input_linear = nn.Linear(input_size, 3*output_size, bias=(not layer_norm))
        self.layer_norm = layer_norm
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        if self.layer_norm:
            self.preact_ln = LayerNorm(3 * output_size)
            #self.x_f_ln = LayerNorm(output_size // 2)
            #self.x_b_ln = LayerNorm(output_size // 2)
            #self.f_g_ln = LayerNorm(output_size // 2)
            #self.b_g_ln = LayerNorm(output_size // 2)
            #self.highway_ln = LayerNorm(output_size)

    def initialize_parameters(self, param_init):
        self.preact_ln.initialize_parameters(param_init)

    def forward(self, input):
        pre_act = self.input_linear(self.dropout(input))
        #h_gate = pre_act[:, :, 2*self.output_size:]
        #gf, gb, x_f, x_b = pre_act[:, :, :2*self.output_size].split(self.output_size // 2, dim=-1)
        if self.layer_norm:
            pre_act = self.preact_ln(pre_act)
            #x_f = self.x_f_ln(x_f)
            #x_b = self.x_b_ln(x_b)
            #gf = self.f_g_ln(gf)
            #gb = self.b_g_ln(gb)
            #h_gate = self.highway_ln(h_gate)
        h_gate = pre_act[:, :, 2*self.output_size:]
        g, x = pre_act[:, :, :2*self.output_size].split(self.output_size, dim=-1)
        gf, gb = F.sigmoid(g).split(self.output_size // 2, dim=-1)
        x_f, x_b = x.split(self.output_size // 2, dim=-1)
        h_gate = F.sigmoid(h_gate)
        h_f_pre = gf * x_f
        h_b_pre = gb * x_b

        h_i_f = Variable(h_f_pre.data.new(gf[0].size()).zero_(), requires_grad=False)
        h_i_b = Variable(h_f_pre.data.new(gf[0].size()).zero_(), requires_grad=False)

        h_f, h_b = [], []
        for i in range(input.size(0)):
            h_i_f = (1. - gf[i]) * h_i_f + h_f_pre[i]
            h_i_b = (1. - gb[-(i+1)]) * h_i_b + h_b_pre[-(i+1)]
            h_f += [h_i_f]
            h_b += [h_i_b]

        h = torch.cat([torch.stack(h_f), torch.stack(h_b[::-1])], dim=-1)

        output = (1. - h_gate) * h + input * h_gate

        return output


class SRU(nn.Module):
  def __init__(self, input_size, output_size, dropout):
    super(SRU, self).__init__()
    self.linear_in = nn.Linear(input_size, 3 * output_size)
    if input_size != output_size:
      self.reduce = nn.Linear(input_size, output_size)
    self.input_size = input_size
    self.output_size = output_size
    self.dropout = nn.Dropout(dropout)
  
  def initialize_parameters(self, param_init):
    pass
  
  def forward(self, prev_layer, hidden):
    """
    :param prev_layer: targetL x batch x output_size
    :param hidden: batch x output_size
    :return:
    """
    
    # targetL x batch x output_size
    preact = self.linear_in(self.dropout(prev_layer))
    
    prev_layer_t = preact[:, :, :self.output_size]
    z, h_gate = F.sigmoid(preact[:, :, self.output_size:]).split(self.output_size, dim=-1)
    
    ss = []
    for i in range(prev_layer.size(0)):
      s = (1 - z[i]) * hidden + z[i] * prev_layer_t[i]
      # targetL x batch x output_size
      ss += [s]
      # batch x output_size
      hidden = s
    
    # (targetL x batch) x output_size
    out = torch.stack(ss)
    if self.input_size != self.output_size:
      prev_layer = self.reduce(self.dropout(prev_layer))
    
    out = (1. - h_gate) * out + h_gate * prev_layer
    
    return out, hidden
