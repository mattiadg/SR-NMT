"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Normalization import LayerNorm

def getAttention(attention_type):
    attns = {'dot': DotAttention,
             'mlp': MLPAttentionGRU,
             }

    if attention_type not in attns:
        raise NotImplementedError(attention_type)

    return attns[attention_type]


class DotAttention(nn.Module):
    def __init__(self, dim, enc_dim=None, layer_norm=False, activ='tanh'):
        super(DotAttention, self).__init__()
        self.mask = None
        if not enc_dim:
            enc_dim = dim
        out_dim = dim
        self.linear_in = nn.Linear(dim, out_dim, bias=False)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln_in = LayerNorm(dim)
    
    def applyMask(self, mask):
        self.mask = mask
    
    def initialize_parameters(self, param_init):
        pass
    
    def forward(self, input, context, values):
        """
        input: targetL x batch x dim
        context: batch x sourceL x dim
        """
        batch, sourceL, dim = context.size()
        targetT = self.ln_in(self.linear_in(input.transpose(0, 1)))  # batch x targetL x dim
        context = context.transpose(1, 2)  # batch x dim x sourceL
        # Get attention
        attn = torch.bmm(targetT, context)  # batch x targetL x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, sourceL))  # (batch x targetL) x sourceL
        attn3 = attn.view(batch, -1, sourceL)  # batch x targetL x sourceL
        weightedContext = torch.bmm(attn3, values).transpose(0, 1)  # targetL x batch x dim
        
        return weightedContext, attn


class MLPAttention(nn.Module):
    def __init__(self, dim, layer_norm=False, activ='tanh'):
        super(MLPAttention, self).__init__()
        self.dim = dim
        self.v = nn.Linear(self.dim, 1)
        self.combine_hid = nn.Linear(self.dim, self.dim)
        #self.combine_ctx = nn.Linear(self.dim, self.dim)
        self.mask = None
        self.activ = getattr(F, activ)
        self.layer_norm = layer_norm
        if layer_norm:
            #self.ctx_ln = LayerNorm(dim)
            self.hidden_ln = LayerNorm(dim)

    def applyMask(self, mask):
        self.mask = mask

    def initialize_parameters(self, param_init):
        pass


    def forward(self, input, context, values):
        """
        input: targetL x batch x dim
        context: batch x sourceL x dim
        values: batch x sourceL x dim

        Output:

        output: batch x hidden_size
        w: batch x sourceL
        """
        targetL = input.size(0)
        output_size = input.size(2)
        sourceL = context.size(1)
        batch_size = input.size(1)

        # targetL x batch x dim
        input = self.combine_hid(input)
        # (targetL x batch) x dim
        #context = self.combine_ctx(context)
        if self.layer_norm:
            input = self.hidden_ln(input)
            #context = self.ctx_ln(context)

        # batch x (sourceL x targetL) x dim
        context = context.repeat(1, targetL, 1)

        # batch x targetL x dim -> batch x (targetL x sourceL) x dim
        input = input.transpose(0, 1).repeat(1, 1, sourceL).contiguous().view(batch_size, -1, output_size)
        #context = context.view(batch_size, -1, output_size)
        # batch x (targetL x sourceL) x dim
        combined = self.activ(input + context)

        # batch x (targetL x sourceL) x 1
        attn = self.v(combined)

        # (batch_size x targetL) x sourceL
        attn = attn.contiguous().view(batch_size * targetL, sourceL)

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        # (batch_size x targetL) x sourceL
        attn = F.softmax(attn)

        # batch_size x targetL x sourceL
        attn3 = attn.contiguous().view(batch_size, targetL, sourceL)

        # batch x targetL x dim -> targetL x batch x dim
        weightedContext = torch.bmm(attn3, values).transpose(0, 1)

        return weightedContext, attn

class MLPAttentionGRU(nn.Module):
     def __init__(self, dim, layer_norm=False, activ='tanh'):
         super(MLPAttentionGRU, self).__init__()
         self.dim = dim
         self.v = nn.Linear(self.dim, 1)
         self.combine_hid = nn.Linear(self.dim, self.dim)
         # self.combine_ctx = nn.Linear(self.dim, self.dim)
         self.mask = None
         self.activ = getattr(F, activ)
         self.layer_norm = layer_norm
         if layer_norm:
             # self.ctx_ln = LayerNorm(dim)
             self.hidden_ln = LayerNorm(dim)

     def applyMask(self, mask):
         self.mask = mask

     def initialize_parameters(self, param_init):
         pass

     def forward(self, input, context, values):
         """
         input: batch x dim
         context: batch x sourceL x dim
         values: batch x sourceL x dim

         Output:

         output: batch x hidden_size
         w: batch x sourceL
         """
         sourceL = context.size(1)
         batch_size = input.size(0)

         # batch x dim
         input = self.combine_hid(input)

         if self.layer_norm:
             input = self.hidden_ln(input)

         # batch x sourceL x dim
         input = input.unsqueeze(1).expand_as(context)
         # batch x sourceL x dim
         combined = self.activ(input + context)

         # batch x sourceL x 1
         attn = self.v(combined)

         # batch_size x sourceL
         attn = attn.view(batch_size, sourceL)

         if self.mask is not None:
             attn.data.masked_fill_(self.mask, -float('inf'))

         # batch_size x sourceL
         attn = F.softmax(attn)

         # batch_size x 1 x sourceL
         attn3 = attn.unsqueeze(1)

         # batch x dim
         weightedContext = torch.bmm(attn3, values).squeeze(1)

         return weightedContext, attn



class SelfAttention(nn.Module):
    def __init__(self, k_size, q_size, v_size, out_size):
        super(SelfAttention, self).__init__()
        self.linearK = nn.Linear(v_size, out_size)
        self.linearQ = nn.Linear(q_size, out_size)
        self.linearV = nn.Linear(v_size, out_size)
        self.dim = out_size
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, values):
        """
        input: batch x targetL x dim
        context: batch x sourceL x dim
        values: batch x sourceL x dim
        """
        K = self.linearK(input)                                                           # batch x targetL x out_size
        Q = self.linearQ(context)                                                         # batch x sourceL x out_size
        V = self.linearV(values)                                                          # batch x sourceL x out_size

        dot_prod = K.bmm(Q.transpose(1, 2)) * (1 / np.sqrt(self.dim))                     # batch x targetL x sourceL

        attn = dot_prod.sum(dim=1, keepdim=False)                                         # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn)                                                            # batch x sourceL
        attn3 = attn.unsqueeze(2)                                                         # batch x sourceL x 1
        weightedContext = V * attn3                                                       # batch x sourceL x out_size

        return weightedContext, attn
