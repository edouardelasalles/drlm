import math
from collections import OrderedDict

import yaml

import torch.nn.functional as F

from drlm import DynamicRecurrentLanguageModel


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
    ntkns = []
    for t, loader in loaders.items():
        nll, ppl, ntkn = evaluate_lm_at_t(model, loader, opt)
        nlls.append(nll)
        ppls.append((t, ppl))
        ntkns.append(ntkn)
    results = [
        ('micro', perplexity(sum(nlls) / sum(ntkns))),
        ('macro', sum(perplexity(nll / ntkn) for nll, ntkn in zip(nlls, ntkns)) / len(nlls)),
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
        ntkn += target.ne(opt.padding_idx).sum().item()
        # forward
        output = model.evaluate(text, timestep)
        # eval
        nll += F.cross_entropy(output.view(-1, opt.ntoken), target.view(-1),
                               ignore_index=opt.padding_idx, reduction='sum').item()
    return nll, perplexity(nll / ntkn), ntkn


def lm_factory(opt):
    if opt.model in ('drlm', 'drlm-id'):
        return DynamicRecurrentLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute,
                                             opt.dropouti, opt.dropoutl, opt.dropouth, opt.dropouto, opt.tied_weights,
                                             opt.nts, opt.nzt, opt.nhid_zt, opt.nlayers_zt, opt.learn_transition,
                                             opt.padding_idx, opt.nwords)
    else:
        raise ValueError('No model named `{}`'.format(opt.model))


def get_lm_optimizers(model, opt):
    if opt.model in ('drlm', 'drlm-id'):
        return model.get_optimizers(opt.lr, opt.wd_lm)
    else:
        raise ValueError('No model named `{}`'.format(opt.model))


def get_lr(optimizers, opt):
    if opt.model in ('drlm', 'drlm-id'):
        return optimizers['adam'].param_groups[0]['lr']
