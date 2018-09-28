import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.rnn import LSTM
from modules.mlp import MLP
from modules.embedding import Embedding
from utils import identity


class DynamicRecurrentLanguageModel(nn.Module):
    def __init__(self, ntoken, nwe, nhid_rnn, nlayers_rnn, dropoute, dropouti, dropoutl, dropouth, dropouto, tied_weights,
                 nts, nzt, nhid_zt, nlayers_zt, res, learn_transition, padding_idx, nwords):
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
        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.lstm = LSTM(nwe + self.nzt, nhid_rnn, nhid_rnn, nlayers_rnn, 0., dropoutl, dropouth, dropouto)
        self.decoder = nn.Linear(nhid_rnn, ntoken)
        self.decoder.bias.data.zero_()
        if self.tied_weights:
            self.decoder.weight = self.word_embedding.weight
        # temporal modules
        self.z0 = nn.Parameter(torch.Tensor(1, self.nzt).uniform_(-0.1, 0.1))
        if self.learn_transition:
            assert res is not None
            self.transition_function = MLP(self.nzt, nhid_zt, self.nzt, nlayers_zt, 0)
            self.res = res
        else:
            assert nhid_zt == 0 and nlayers_zt == 0 and res is None
            self.transition_function = identity
        self.q_mu = nn.Parameter(torch.Tensor(nts, self.nzt).fill_(0))
        self.q_logvar = nn.Parameter(torch.Tensor(nts, self.nzt).fill_(0))
        self.p_logvar = nn.Parameter(torch.Tensor(1).fill_(0))

    def _rsample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def transition(self, zt):
        zt_next = self.transition_function(zt)
        if self.res:
            zt_next = zt + zt_next
        return torch.tanh(zt_next)

    def infer_zt(self):
        q_mu, q_logvar = torch.tanh(self.q_mu), self.q_logvar
        return self._rsample(q_mu, q_logvar), (q_mu, q_logvar)

    def predict_zt(self, k, z0=None):
        zt = z0.unsqueeze(0) if z0 is not None else self.z0
        states = []
        for _ in range(k):
            zt = self.transition(zt)
            states.append(zt)
        return torch.cat(states)

    def forward(self, text, zt, hidden=None):
        emb = self.word_embedding(text)
        lstm_input = torch.cat((emb, zt.unsqueeze(0).expand(*text.shape, self.nzt)), -1)
        output, hidden = self.lstm(lstm_input, hidden)
        return self.decoder(output), hidden

    def get_parameters(self, wd_lm, wd_transition):
        params_lm = list(self.word_embedding.parameters()) + list(self.lstm.parameters())
        if self.tied_weights:
            params_lm.append(self.decoder.bias)
        else:
            params_lm += list(self.decoder.parameters())
        params_distrib = [self.q_mu, self.q_logvar, self.p_logvar, self.z0]
        params = [
            {'params': params_lm, 'weight_decay': wd_lm, 'betas': (0.0, 0.999)},
            {'params': params_distrib, 'weight_decay': 0., 'betas': (0.9, 0.999)},
        ]
        if self.learn_transition:
            params.append({'params': self.transition_function.parameters(), 'weight_decay': wd_transition, 'betas': (0.9, 0.999)})
        # test number paramters
        nparams = sum(p.nelement() for param_group in params for p in param_group['params'])
        true_nparams = sum([p.nelement() for p in self.parameters()])
        assert true_nparams == nparams
        return params

    def closure(self, text, target, timestep):
        # latent states
        zt, (q_mu, q_logvar) = self.infer_zt()
        p_mu = self.transition(torch.cat((self.z0, zt[:-1])))
        # language model
        output, _ = self.forward(text, zt[timestep])
        # nll
        nll = F.cross_entropy(output.view(-1, self.ntoken), target.view(-1), ignore_index=self.padding_idx)
        # kl
        kld = self.p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu)**2) / self.p_logvar.exp() - 1
        kld = 0.5 * kld.sum()
        # rescaled elbo
        elbo = nll + (1 / self.nwords) * kld
        return elbo

    def evaluate(self, text, timestep):
        nll = 0
        zt = self.predict_zt(max(timestep) + 1)
        output, _ = self.forward(text, zt[timestep])
        return output
