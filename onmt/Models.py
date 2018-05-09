import torch.nn as nn
from torch.autograd import Variable
from .Encoders import Encoder


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def load_pretrained_vectors(self, opt):
        self.encoder.load_pretrained_vectors(opt)
        self.decoder.load_pretrained_vectors(opt)

    def initialize_parameters(self, param_init):
        self.encoder.initialize_parameters(param_init)
        self.decoder.initialize_parameters(param_init)

    def brnn_merge_concat(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context, emb = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        if isinstance(self.encoder, Encoder):
            if isinstance(enc_hidden, tuple):
                enc_hidden = tuple(self.brnn_merge_concat(enc_hidden[i])
                                   for i in range(len(enc_hidden)))
            else:
                enc_hidden = self.brnn_merge_concat(enc_hidden)
                if enc_hidden.size(0) < self.decoder.layers:
                    enc_hidden = enc_hidden.repeat(self.decoder.layers, 1, 1)
        else:
            enc_hidden = Variable(enc_hidden.data.new(*enc_hidden.size()).zero_(), requires_grad=False)

        #self.decoder.mask_attention(src[0])
        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden,
                                              context, init_output)
        return out
