import math

import yaml

import torch.nn.functional as F

from lstm_lm import LSTMLanguageModel
from drlm import DynamicRecurrentLanguageModel


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path):
    with open(path, 'r') as f:
        return DotDict(yaml.load(f))


def perplexity(nll):
    try:
        return math.exp(nll)
    except OverflowError:
        return float('inf')


def evaluate_lm(model, dataloader, opt):
    nll = 0
    ntkn = 0
    for batch in dataloader:
        # inputs
        text = batch.text[0][:-1]
        target = batch.text[0][1:]
        timesteps = batch.timestep
        # forward
        output = model.evaluate(text, timesteps)
        # eval
        nll += F.cross_entropy(output.view(-1, opt.ntoken), target.view(-1),
                               ignore_index=opt.padding_idx, reduction='sum').item()
        ntkn += target.ne(opt.padding_idx).sum().item()
    nll /= ntkn
    return nll, perplexity(nll), ntkn


def lm_factory(opt):
    if opt.model == 'lstm':
        return LSTMLanguageModel(opt.ntoken, opt.nwe, opt.nhid, opt.nlayers, opt.dropoute, opt.dropouti, opt.dropoutl,
                                 opt.dropouth, opt.dropouto, opt.tie_weights, opt.padding_idx)
    elif opt.model in ('drlm', 'drlm-id'):
        return DynamicRecurrentLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute,
                                             opt.dropouti, opt.dropoutl, opt.dropouth, opt.dropouto, opt.tie_weights,
                                             opt.nts_train, opt.nzt, opt.nhid_zt, opt.nlayers_zt, opt.res_zt,
                                             opt.learn_transition, opt.padding_idx, opt.nwords_train)
    else:
        raise ValueError('No model named `{}`'.format(opt.model))


def get_lm_parameters(model, opt):
    if opt.model == 'lstm':
        return model.get_parameters(opt.wd)
    elif opt.model in ('drlm', 'drlm-id'):
        return model.get_parameters(opt.wd, opt.wd_t)
    else:
        raise ValueError('No model named `{}`'.format(opt.model))
