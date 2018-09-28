import torch
import torch.nn as nn

from .dropout import LockedDropout, WeightDrop


# Code inspired from AWD-LSTM :
# Regularizing and Optimizing LSTM Language Models.
# Merity et al. ICLR 2018
# https://github.com/salesforce/awd-lstm-lm/


class LSTM(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, dropouti, dropoutl, dropouth, dropouto):
        super(LSTM, self).__init__()
        # assert
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.nlayers = nlayers
        self.idrop = LockedDropout(dropouti)
        self.ldrop = LockedDropout(dropoutl)
        self.odrop = LockedDropout(dropouto)
        # LSTM
        self.rnns = [nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nout, 1)
                     for l in range(nlayers)]
        self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=dropouth) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)

    def forward(self, input, hidden=None):
        output = input
        new_hidden = []
        outputs = []
        # forward through layers
        for l, rnn in enumerate(self.rnns):
            input_l = output
            h_n = None if hidden is None else (hidden[0][l], hidden[1][l])
            # dropout
            input_l_droped = self.idrop(input_l) if l == 0 else self.ldrop(input_l)
            if l > 0:
                outputs.append(input_l_droped)
            # forward
            raw_output, new_h = rnn(input_l_droped, h_n)
            new_hidden.append(new_h)
            output = raw_output
        raw_output = output
        output = self.odrop(output)
        outputs.append(output)
        h_n = [h_n_l for h_n_l, _ in new_hidden]
        c_n = [c_n_l for _, c_n_l in new_hidden]
        hidden = (h_n, c_n)
        return output, hidden
