import math
from collections import OrderedDict

import yaml

import torch.nn.functional as F

from lstm_lm import LSTMLanguageModel
from drlm import DynamicRecurrentLanguageModel
from dwe import DynamicWordEmbeddingLangaugeModel
from dt import DiffTimeLanguageModel


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def perplexity(nll):
    try:
        return math.exp(nll)
    except OverflowError:
        return float('inf')


def load_config(path):
    with open(path, 'r') as f:
        return DotDict(yaml.load(f))


def evaluate_lm(model, loaders, opt):
    assert not model.training
    nlls = []
    ppls = []
    ntkn_test = 0
    for t, loader in loaders.items():
        nll, ppl, ntkn = evaluate_lm_at_t(model, loader, opt)
        nlls.append(nll)
        ppls.append((t, ppl))
        ntkn_test += ntkn
    results = [
        ('micro', perplexity(sum(nlls) / ntkn_test)),
        ('macro', sum(perplexity(nll) for nll in nlls) / len(nlls)),
    ]
    return OrderedDict(results + ppls)


def evaluate_lm_at_t(model, loader_t, opt):
    assert not model.training
    nll = 0
    ntkn = 0
    for batch in loader_t:
        # inputs
        text = batch.text[0][:-1]
        target = batch.text[0][1:]
        timesteps = batch.timestep
        timestep = batch.timestep.unique().item()
        # forward
        output = model.evaluate(text, timestep)
        # eval
        nll += F.cross_entropy(output.view(-1, opt.ntoken), target.view(-1),
                               ignore_index=opt.padding_idx, reduction='sum').item()
        ntkn += target.ne(opt.padding_idx).sum().item()
    nll /= ntkn
    return nll, perplexity(nll), ntkn


def lm_factory(opt):
    if opt.model == 'lstm':
        return LSTMLanguageModel(opt.ntoken, opt.nwe, opt.nhid, opt.nlayers, opt.dropoute, opt.dropouti, opt.dropoutl,
                                 opt.dropouth, opt.dropouto, opt.tied_weights, opt.padding_idx)
    elif opt.model in ('drlm', 'drlm-id'):
        return DynamicRecurrentLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute,
                                             opt.dropouti, opt.dropoutl, opt.dropouth, opt.dropouto, opt.tied_weights,
                                             opt.nts, opt.nzt, opt.nhid_zt, opt.nlayers_zt, opt.res_zt,
                                             opt.learn_transition, opt.padding_idx, opt.nwords)
    elif opt.model == 'dwe':
        return DynamicWordEmbeddingLangaugeModel(opt.ntoken, opt.nwe, opt.nhid, opt.nlayers, opt.dropoute,
                                                 opt.dropouti, opt.dropoutl, opt.dropouth, opt.dropouto, opt.nts,
                                                 opt.sigma_0, opt.sigma_t, opt.padding_idx, opt.nwords)
    elif opt.model == 'dt':
        return DiffTimeLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute, opt.dropouti,
                                     opt.dropoutl, opt.dropouth, opt.dropouto, opt.tie_weights, opt.nts, opt.nhid_t,
                                     opt.padding_idx)
    else:
        raise ValueError('No model named `{}`'.format(opt.model))


def get_lm_optimizers(model, opt):
    if opt.model in ('lstm', 'dwe', 'dt'):
        return model.get_optimizers(opt.lr, opt.wd)
    elif opt.model in ('drlm', 'drlm-id'):
        return model.get_optimizers(opt.lr, opt.wd_lm, opt.wd_t)
    else:
        raise ValueError('No model named `{}`'.format(opt.model))


def get_lr(optimizers, opt):
    if opt.model in ('lstm', 'drlm', 'drlm-id', 'dt'):
        return optimizers['adam'].param_groups[0]['lr']
    elif opt.model == 'dwe':
        return optimizers['adam_lm'].param_groups[0]['lr']
