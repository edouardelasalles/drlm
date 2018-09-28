import math


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


def identity(input):
    return input
