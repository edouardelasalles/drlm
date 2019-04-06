import torch
import torch.nn as nn
import torch.nn.functional as F

from module.rnn import LSTM
from module.mlp import MLP
from module.embedding import Embedding


class DynamicRecurrentLanguageModel(nn.Module):
    def __init__(self, ntoken, nwe, nhid_rnn, nlayers_rnn, dropoute, dropouti, dropoutl, dropouth, dropouto, tied_weights,
                 nts, nzt, nhid_zt, nlayers_zt, learn_transition, padding_idx, nwords):
        super(DynamicRecurrentLanguageModel, self).__init__()
        # attributes
        self.ntoken = ntoken
        self.nts = nts
        self.tied_weights = tied_weights
        self.learn_transition = learn_transition
        self.nzt = nzt
        self.nwords = nwords
        self.padding_idx = padding_idx
        # language model modules
        self.word_embedding = Embedding(ntoken, nwe, dropout=dropouti, dropoute=dropoute, padding_idx=self.padding_idx)
        self.lstm = LSTM(nwe + self.nzt, nhid_rnn, nhid_rnn, nlayers_rnn, 0., dropoutl, dropouth, dropouto)
        self.decoder = nn.Linear(nhid_rnn, ntoken)
        if self.tied_weights:
            self.decoder.weight = self.word_embedding.weight
        # temporal modules
        self.z0 = nn.Parameter(torch.Tensor(1, self.nzt).uniform_(-0.1, 0.1))
        if self.learn_transition:
            self.transition_function = MLP(self.nzt, nhid_zt, self.nzt, nlayers_zt, 0)
        else:
            assert nhid_zt == 0 and nlayers_zt == 0
            self.transition_function = lambda x: x
        self.q_mu = nn.Parameter(torch.Tensor(nts, self.nzt).fill_(0))
        self.q_logvar = nn.Parameter(torch.Tensor(nts, self.nzt).fill_(0))
        self.p_logvar = nn.Parameter(torch.Tensor(1).fill_(0))
        # init
        self._init()

    def _init(self):
        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.word_embedding.weight.data[self.padding_idx] = 0
        self.decoder.bias.data.zero_()

    def _rsample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def transition(self, zt):
        zt_next = self.transition_function(zt)
        if self.learn_transition:
            res = zt_next
            zt_next = zt + zt_next
            return zt_next, res
        return zt_next, None

    def infer_zt(self):
        q_mu, q_logvar = self.q_mu, self.q_logvar
        return self._rsample(q_mu, q_logvar), (q_mu, q_logvar)

    def predict_zt(self, k, z0=None):
        zt = z0.unsqueeze(0) if z0 is not None else self.z0
        states = []
        res = []
        for _ in range(k):
            zt, r = self.transition(zt)
            states.append(zt)
            res.append(r)
        return torch.cat(states), torch.cat(states)

    def forward(self, text, zt, hidden=None):
        emb = self.word_embedding(text)
        lstm_input = torch.cat((emb, zt.unsqueeze(0).expand(*text.shape, self.nzt)), -1)
        output, hidden = self.lstm(lstm_input, hidden)
        return self.decoder(output), hidden

    def closure(self, text, target, timestep, optimizers, config):
        optimizer = optimizers['adam']
        optimizer.zero_grad()
        # latent states
        zt, (q_mu, q_logvar) = self.infer_zt()
        p_mu, res = self.transition(torch.cat((self.z0, zt[:-1])))
        # language model
        output, _ = self.forward(text, zt[timestep])
        # nll
        nll = F.cross_entropy(output.view(-1, self.ntoken), target.view(-1), ignore_index=self.padding_idx)
        # kl
        kld = self.p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu)**2) / self.p_logvar.exp() - 1
        kld = 0.5 * kld.sum()
        # rescaled elbo
        loss = nll + (1 / self.nwords) * config['beta'] * kld
        # regularisation
        if config['wd_q'] > 0:
            loss += config['wd_q'] * q_mu.pow(2).mean()
        if self.learn_transition:
            if config['wd_res'] > 0:
                loss += config['wd_res'] * res.pow(2).mean()
            if config['wd_inertia'] > 0:
                loss += config['wd_inertia'] * res[:-1].sub(res[1:]).pow(2).mean()
        # backward
        loss.backward()
        # step
        optimizer.step()
        return loss.item()

    def evaluate(self, text, t):
        z, _ = self.predict_zt(t + 1)
        zt = z[t].unsqueeze(0).expand(text.shape[1], self.nzt)
        output, _ = self.forward(text, zt)
        return output

    def get_optimizers(self, lr, wd):
        wd_params = []
        params = []
        for name, p in self.named_parameters():
            if name.split('.')[0] in ('word_embedding', 'lstm', 'decoder'):
                wd_params.append(p)
            else:
                params.append(p)
        optimizer = torch.optim.Adam([
            {'params': wd_params, 'weight_decay': wd},
            {'params': params},
        ], lr=lr)
        return {'adam': optimizer}
