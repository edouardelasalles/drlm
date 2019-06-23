import yaml

from lstm import LSTMLanguageModel
from drlm import DynamicRecurrentLanguageModel


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path):
    with open(path, 'r') as f:
        return DotDict(yaml.safe_load(f))


def lm_factory(opt):
    if opt.model in ('drlm', 'drlm-id'):
        return DynamicRecurrentLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute,
                                             opt.dropouti, opt.dropoutl, opt.dropouth, opt.dropouto, opt.tied_weights,
                                             opt.nts, opt.nzt, opt.nhid_zt, opt.nlayers_zt, opt.learn_transition,
                                             opt.padding_idx, opt.nwords)
    if opt.model == 'lstm':
        return LSTMLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute, opt.dropouti,
                                 opt.dropoutl, opt.dropouth, opt.dropouto, opt.tied_weights, opt.padding_idx,
                                 opt.nwords)
    raise ValueError('No model named `{}`'.format(opt.model))


def get_lm_optimizers(model, opt):
    return model.get_optimizers(opt.lr, opt.wd_lm)


def get_lr(optimizers, opt):
    return optimizers['adam'].param_groups[0]['lr']
