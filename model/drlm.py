from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module.rnn import LSTM
from .module.mlp import MLP
from .module.embedding import Embedding
from .module.utils import init_weight
from evaluate import perplexity


class DynamicRecurrentLanguageModel(nn.Module):
    def __init__(self, ntoken, nwe, nhid_rnn, nlayers_rnn, dropoute, dropouti, dropoutl, dropouth, dropouto,
                 tied_weights, nts, nzt, nhid_zt, nlayers_zt, learn_transition, padding_idx, nwords):
        super().__init__()
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
        self.q_mu = nn.Parameter(torch.zeros(nts, self.nzt))
        self.q_logvar = nn.Parameter(torch.zeros(nts, self.nzt))
        self.p_logvar = nn.Parameter(torch.zeros(1, self.nzt))
        # init
        self._init()

    def _init(self):
        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.word_embedding.weight.data[self.padding_idx] = 0
        self.decoder.bias.data.zero_()
        if self.learn_transition:
            init_fn = partial(init_weight, init_type='orthogonal', init_gain=0.02)
            self.transition_function.apply(init_fn)

    def _rsample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def transition(self, zt):
        if self.learn_transition:
            res = self.transition_function(zt)
            zt_next = zt + res
            return zt_next, res
        return zt, None

    def infer_zt(self):
        q_mu, q_logvar = self.q_mu, self.q_logvar
        return self._rsample(q_mu, q_logvar), (q_mu, q_logvar)

    def predict_zt(self, z0, nt):
        zt = z0
        states = []
        res = []
        for t in range(nt):
            zt, r = self.transition(zt)
            if r is not None:
                res.append(r)
            states.append(zt)
        states = torch.stack(states)
        res = torch.stack(res) if len(res) > 0 else res
        return states, res

    def forward(self, text, zt, hidden=None):
        emb = self.word_embedding(text)
        lstm_input = torch.cat((emb, zt.unsqueeze(0).expand(*text.shape, self.nzt)), -1)
        output, hidden = self.lstm(lstm_input, hidden)
        return self.decoder(output), hidden

    def closure(self, text, target, timestep, optimizer, config):
        optimizer.zero_grad()
        # latent states
        zt, (q_mu, q_logvar) = self.infer_zt()
        p_mu, res = self.transition(torch.cat((self.z0, zt[:-1])))
        # language model
        output, _ = self.forward(text, zt[timestep])
        # nll
        nll = F.cross_entropy(output.view(-1, self.ntoken), target.view(-1), ignore_index=self.padding_idx)
        # kl
        p_logvar = self.p_logvar.expand_as(q_logvar)
        kld = p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu)**2) / p_logvar.exp() - 1
        kld = (1 / self.nwords) * 0.5 * kld.sum()
        # rescaled elbo
        loss = nll + config['beta'] * kld
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
        # log
        log = {}
        log['loss'] = loss.item()
        log['nll'] = nll.item()
        log['kld'] = kld.item()
        log['elbo'] = log['nll'] + log['kld']
        log['ppl'] = perplexity(log['nll'])
        return log

    def evaluate(self, text, t, from_z0=False):
        if from_z0:
            z_pred, _ = self.predict_zt(self.z0.squeeze(0), t + 1)
            assert t == z_pred.shape[0] - 1
            zt = z_pred[-1]
        else:
            if t < self.nts:
                zt = self.q_mu[t]
            else:
                z_pred, _ = self.predict_zt(self.q_mu[-1], t + 1 - self.nts)
                zt = z_pred[-1]
        zt = zt.unsqueeze(0).expand(text.shape[1], self.nzt)
        output, _ = self.forward(text, zt)
        return output

    def get_optimizer(self, lr, wd):
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
        return optimizer
