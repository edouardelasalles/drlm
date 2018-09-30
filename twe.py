import torch
import torch.nn as nn
import torch.nn.functional as F

from module.rnn import LSTM


class TemporalWordEmbeddingLanguageModel(nn.Module):
    def __init__(self, nts, ntoken, nwe, nhid, nlayers, dropoute, dropouti, dropoutl, dropouth, dropouto, padding_idx):
        super(TemporalWordEmbeddingLanguageModel, self).__init__()
        # attributes
        self.ntoken = ntoken
        self.nwe = nwe
        self.dropoute = dropoute
        self.nts = nts
        self.padding_idx = padding_idx
        # emebedings
        self.U = nn.Embedding(self.ntoken, self.nts * self.nwe, sparse=True, padding_idx=self.padding_idx)
        self.U.weight.data.uniform_(-0.1, 0.1)
        # lstm
        self.lstm = LSTM(self.nwe, nhid, nhid, nlayers, dropouti, dropoutl, dropouth, dropouto)
        # decoder
        self.decoder = nn.Linear(nhid, self.ntoken)
        self.decoder.bias.data.zero_()

    def forward(self, embeddings, hidden=None):
        output, hidden = self.lstm(embeddings, hidden)
        return self.decoder(output), hidden

    def closure(self, text, target, timestep, optimizers):
        optimizer_lm = optimizers['adam_lm']
        optimizer_we = optimizers['adam_twe']
        optimizer_lm.zero_grad()
        optimizer_we.zero_grad()
        # flatten texts
        text_flat = text.view(-1)
        timestep_flat = timestep.unsqueeze(0).expand_as(text).contiguous().view(-1)
        # get unique words in text
        words = text_flat.unique()
        idx_change = text.new_zeros(self.ntoken)
        idx_change[words] = torch.arange(len(words), device=words.device, dtype=words.dtype)
        # select embs
        U = self.U(words).view(len(words), self.nts, self.nwe).transpose(0, 1).contiguous()
        emb_raw = U[timestep_flat, idx_change[text_flat]]
        # dropout
        if self.training and self.dropoute > 0.:
            mask = emb_raw.new(self.ntoken).bernoulli_(1 - self.dropoute) / (1 - self.dropoute)
            mask = mask[text_flat].unsqueeze(1).expand_as(emb_raw)
            emb = mask * emb_raw
        else:
            emb = emb_raw
        emb = emb.view(*text.shape, self.nwe)
        # lm
        output, _ = self.forward(emb)
        # nll
        nll = F.cross_entropy(output.view(-1, self.ntoken), target.view(-1), ignore_index=self.padding_idx)
        # backward
        nll.backward()
        # step
        optimizer_lm.step()
        optimizer_we.step()
        return nll.item()

    def evaluate(self, text, t):
        t = text.new(1).fill_(min(t, self.nts - 1))
        emb = self.U(text).view(*text.shape, self.nts, self.nwe)
        emb_t = emb.index_select(text.ndimension(), t).squeeze(text.ndimension())
        output, _ = self.forward(emb_t)
        return output

    def get_optimizers(self, lr, wd):
        # language model
        lstm_params = [{'params': self.lstm.parameters()}, {'params': self.decoder.parameters()}]
        optim_lm = torch.optim.Adam(lstm_params, lr=lr, betas=(0.0, 0.999), eps=1e-9, weight_decay=wd)
        # dynamic word embeddings
        dwe_params = [{'params': self.U.parameters()}]
        optim_dwe = torch.optim.SparseAdam(dwe_params, lr=lr, betas=(0.9, 0.999), eps=1e-9)
        return {'adam_twe': optim_dwe, 'adam_lm': optim_lm}
