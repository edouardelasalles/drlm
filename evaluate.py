import math
from collections import OrderedDict

import torch.nn.functional as F


def perplexity(nll):
    try:
        return math.exp(nll)
    except OverflowError:
        return float('inf')


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
    micro_ppl = perplexity(sum(nlls) / sum(ntkns))
    results = [
        ('micro', micro_ppl),
        ('macro', sum(perplexity(nll / ntkn) for nll, ntkn in zip(nlls, ntkns)) / len(nlls)),
    ]
    return micro_ppl, OrderedDict(results + ppls)


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
        output = model.evaluate(text, timestep, opt.from_z0)
        # eval
        nll += F.cross_entropy(output.view(-1, opt.ntoken), target.view(-1),
                               ignore_index=opt.padding_idx, reduction='sum').item()
    return nll, perplexity(nll / ntkn), ntkn
