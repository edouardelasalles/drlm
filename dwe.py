import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.rnn import LSTM


class DynamicWordEmbeddingLangaugeModel(nn.Module):
    def __init__(self, ntoken, nwe, nhid, nlayers, dropoute, dropouti, dropoutl, dropouth, dropouto,
                 nts, sigma_0, sigma_t, padding_idx, nwords):
        super(DynamicWordEmbeddingLangaugeModel, self).__init__()
        # attributes
        self.ntoken = ntoken
        self.nwe = nwe
        self.nts = nts
        self.sigma_0 = sigma_0
        self.sigma_t = sigma_t
        self.dropoute = dropoute
        self.padding_idx = padding_idx
        self.nwords = nwords
        self.word_scales = None
        # emebedings
        self.U_mu = nn.Embedding(ntoken, nts * nwe, sparse=True, padding_idx=padding_idx)
        self.U_mu.weight.data.zero_()
        self.U_logvar = nn.Embedding(ntoken, nts * nwe, sparse=True, padding_idx=padding_idx)
        self.U_logvar.weight.data.zero_()
        # lstm
        self.lstm = LSTM(nwe, nhid, nhid, nlayers, dropouti, dropoutl, dropouth, dropouto)
        # decoder
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()

    def _rsample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _compute_prior(self, mu, logvar):
        var_inv = 1 / (logvar.exp() + self.sigma_t)
        var_p = 1 / (var_inv + (1 / self.sigma_0))
        mu_p = var_p * var_inv * mu
        return mu_p, var_p.log()

    def forward(self, embeddings, hidden=None):
        output, hidden = self.lstm(embeddings, hidden)
        return self.decoder(output), hidden

    def closure(self, text, target, timestep, optimizers):
        optimizer_lm = optimizers['adam_lm']
        optimizer_we = optimizers['adam_dwe']
        optimizer_lm.zero_grad()
        optimizer_we.zero_grad()
        # flatten inputs
        text_flat = text.flatten()
        timestep_flat = timestep.unsqueeze(0).expand_as(text).contiguous().flatten()
        # get unique words in text
        words = text_flat.unique()
        idx_change = text.new_zeros(self.ntoken)
        idx_change[words] = torch.arange(len(words), device=words.device, dtype=words.dtype)
        # infer embeddings
        U_mu = self.U_mu(words).view(len(words), self.nts, self.nwe).transpose(0, 1).contiguous()
        U_logvar = self.U_logvar(words).view(len(words), self.nts, self.nwe).transpose(0, 1).contiguous()
        U = self._rsample(U_mu, U_logvar)
        # select embs
        emb = U[timestep_flat, idx_change[text_flat]]
        # dropout
        if self.training and self.dropoute > 0.:
            mask = emb.new(self.ntoken).bernoulli_(1 - self.dropoute) / (1 - self.dropoute)
            mask = mask[text_flat].unsqueeze(1).expand_as(emb)
            emb = mask * emb
        emb = emb.view(*text.shape, self.nwe)
        # lm
        output, _ = self.forward(emb)
        # nll
        nll = F.cross_entropy(output.view(-1, self.ntoken), target.view(-1), ignore_index=self.padding_idx)
        # kl
        U_mu_prev = torch.cat((torch.zeros_like(U_mu[0]).unsqueeze(0), U_mu[:-1]))
        U_logvar_prev = torch.cat((torch.zeros_like(U_logvar[0]).unsqueeze(0).fill_(math.log(self.sigma_0)), U_logvar[:-1]))
        U_mu_prior, U_logvar_prior = self._compute_prior(U_mu_prev, U_logvar_prev)
        kld = U_logvar_prior - U_logvar + (U_logvar.exp() + (U_mu - U_mu_prior)**2) / U_logvar_prior.exp() - 1
        kld = kld.mul(0.5).sum([0, 2]) / self.word_scales[words]
        kld = kld.sum() / self.nwords
        # elbo
        elbo = nll + kld
        # backward
        elbo.backward()
        # step
        optimizer_lm.step()
        optimizer_we.step()
        return elbo.item()

    def evaluate(self, text, t):
        t = text.new(1).fill_(min(t, self.nts - 1))
        emb = self.U_mu(text).view(*text.shape, self.nts, self.nwe)
        emb_t = emb.index_select(text.ndimension(), t).squeeze(text.ndimension())
        output, _ = self.forward(emb_t)
        return output

    def get_optimizers(self, lr, wd):
        # language model
        lstm_params = [{'params': self.lstm.parameters()}, {'params': self.decoder.parameters()}]
        optim_lm = torch.optim.Adam(lstm_params, lr=lr, betas=(0.0, 0.999), eps=1e-9, weight_decay=wd)
        # dynamic word embeddings
        dwe_params = [{'params': self.U_mu.parameters()}, {'params': self.U_logvar.parameters()}]
        optim_dwe = torch.optim.SparseAdam(dwe_params, lr=lr, betas=(0.9, 0.999), eps=1e-9)
        return {'adam_dwe': optim_dwe, 'adam_lm': optim_lm}
