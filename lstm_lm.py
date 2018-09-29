import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.rnn import LSTM
from modules.embedding import Embedding


class LSTMLanguageModel(nn.Module):
    def __init__(self, ntoken, nwe, nhid, nlayers, dropoute, dropouti, dropoutl, dropouth, dropouto,
                 tied_weights, padding_idx):
        super(LSTMLanguageModel, self).__init__()
        # attributes
        self.ntoken = ntoken
        self.tied_weights = tied_weights
        self.padding_idx = padding_idx
        # language model modules
        self.word_embedding = Embedding(ntoken, nwe, dropout=dropouti, dropoute=dropoute, padding_idx=self.padding_idx)
        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.word_embedding.weight.data[self.padding_idx] = 0
        self.lstm = LSTM(nwe, nhid, nhid, nlayers, 0., dropoutl, dropouth, dropouto)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()
        if self.tied_weights:
            self.decoder.weight = self.word_embedding.weight

    def forward(self, text, hidden=None):
        emb = self.word_embedding(text)
        output, hidden = self.lstm(emb, hidden)
        return self.decoder(output), hidden

    def get_parameters(self, wd):
        return [{'params': self.parameters(), 'weight_decay': wd, 'betas': (0.0, 0.999)}]

    def closure(self, text, target, timestep):
        output, _ = self.forward(text)
        nll = F.cross_entropy(output.view(-1, self.ntoken), target.view(-1), ignore_index=self.padding_idx)
        return nll

    def evaluate(self, text, timestep):
        output, _ = self.forward(text)
        return output
