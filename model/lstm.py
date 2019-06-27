from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module.rnn import LSTM
from .module.embedding import Embedding
from evaluate import perplexity


class LSTMLanguageModel(nn.Module):
    def __init__(self, ntoken, nwe, nhid_rnn, nlayers_rnn, dropoute, dropouti, dropoutl, dropouth, dropouto,
                 tied_weights, padding_idx, nwords):
        super().__init__()
        # attributes
        self.ntoken = ntoken
        self.tied_weights = tied_weights
        self.nwords = nwords
        self.padding_idx = padding_idx
        # language model modules
        self.word_embedding = Embedding(ntoken, nwe, dropout=dropouti, dropoute=dropoute, padding_idx=self.padding_idx)
        self.lstm = LSTM(nwe, nhid_rnn, nhid_rnn, nlayers_rnn, 0., dropoutl, dropouth, dropouto)
        self.decoder = nn.Linear(nhid_rnn, ntoken)
        if self.tied_weights:
            self.decoder.weight = self.word_embedding.weight
        # init
        self._init()

    def _init(self):
        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.word_embedding.weight.data[self.padding_idx] = 0
        self.decoder.bias.data.zero_()

    def forward(self, text, hidden=None):
        emb = self.word_embedding(text)
        output, hidden = self.lstm(emb, hidden)
        return self.decoder(output), hidden

    def closure(self, text, target, timestep, optimizer, config):
        optimizer.zero_grad()
        # language model
        output, _ = self.forward(text)
        # nll
        nll = F.cross_entropy(output.view(-1, self.ntoken), target.view(-1), ignore_index=self.padding_idx)
        # rescaled elbo
        loss = nll
        # backward
        loss.backward()
        # step
        optimizer.step()
        # logs
        logs = {}
        logs['loss'] = loss.item()
        logs['ppl'] = perplexity(nll.item())
        return logs

    def evaluate(self, text, *args, **kwargs):
        output, _ = self.forward(text)
        return output

    def get_optimizer(self, lr, wd):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
