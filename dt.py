import torch
import torch.nn as nn
import torch.nn.functional as F

from module.embedding import Embedding
from module.rnn import LSTM


class DiffTimeLanguageModel(nn.Module):
    def __init__(self, ntoken, nwe, nhid_rnn, nlayers_rnn, dropoute, dropouti, dropoutl, dropouth, dropouto, tie_weights,
                 nts, nhid_t, padding_idx):
        super(DiffTimeLanguageModel, self).__init__()
        # attributes
        self.ntoken = ntoken
        self.nwe = nwe
        self.nhid_t = nhid_t
        self.tie_weights = tie_weights
        self.nts = nts
        self.padding_idx = padding_idx
        # emebedings
        self.U = Embedding(self.ntoken, self.nwe, dropout=0., dropoute=dropoute, padding_idx=self.padding_idx)
        self.trans = nn.Linear(self.nwe, self.nhid_t * self.nhid_t)
        self.tc = nn.Sequential(nn.Linear(1, self.nhid_t), nn.Tanh(), nn.Linear(self.nhid_t, self.nhid_t), nn.Tanh())
        self.decoder_we = nn.Linear(self.nhid_t, self.nwe)
        # lstm
        self.lstm = LSTM(self.nwe, nhid_rnn, nhid_rnn, nlayers_rnn, dropouti, dropoutl, dropouth, dropouto)
        # decoder
        self.decoder = nn.Linear(nhid_rnn, self.ntoken)
        self.decoder.bias.data.zero_()
        if self.tie_weights:
            self.decoder.weight = self.U.weight

    def forward(self, text, timestep, hidden=None):
        timestep = timestep.float() / self.nts
        # flatten
        text_flat = text.flatten()
        timestep_flat = timestep.unsqueeze(0).expand_as(text).flatten()
        # embeddings
        embs = self.U(text_flat)
        transfo = self.trans(embs).view(-1, self.nhid_t, self.nhid_t)
        ht = self.tc(timestep_flat.unsqueeze(1))
        vec = transfo.matmul(ht.unsqueeze(2)).squeeze(2)
        emb = self.decoder_we(vec).view(*text.shape, self.nwe)
        # lstm
        output, hidden = self.lstm(emb, hidden)
        return self.decoder(output), hidden

    def get_optimizers(self, lr, wd):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-9, weight_decay=wd)
        return {'adam': optimizer}

    def closure(self, text, target, timestep, optimizers):
        optimizer = optimizers['adam']
        optimizer.zero_grad()
        # forward
        output, _ = self.forward(text, timestep)
        nll = F.cross_entropy(output.view(-1, self.ntoken), target.view(-1), ignore_index=self.padding_idx)
        # backward
        nll.backward()
        # step
        optimizer.step()
        return nll.item()

    def evaluate(self, text, t):
        timestep = text.new(text.shape[1]).fill_(t)
        output, _ = self.forward(text, timestep)
        return output
