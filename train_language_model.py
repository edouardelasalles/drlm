import configargparse
import os
import json
import random
from collections import defaultdict, OrderedDict

from tqdm import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
from torchtext.data import Iterator, BucketIterator, Dataset

from corpus import Corpus
from utils import DotDict, perplexity, load_config, lm_factory, evaluate_lm, get_lm_parameters


def main(opt):
    opt.hostname = os.uname()[1]
    # cudnn
    if opt.device > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.device > -1:
        torch.cuda.manual_seed_all(opt.manualSeed)

    ##################################################################################################################
    # Data
    ##################################################################################################################
    # load config
    data_opt = load_config(os.path.join('configs', opt.corpus, opt.config, 'corpus.yaml'))
    opt = DotDict({**opt, **data_opt})
    # load data
    corpus = Corpus(opt.dataroot)
    # split
    trainset, valset, testset = corpus.split(opt.config, opt.min_freq)
    # dataloaders
    train_loader = Iterator(trainset, opt.batch_size, repeat=False, sort_within_batch=True, device=device)
    val_loader = BucketIterator(valset, opt.batch_size, train=False, device=device)
    ts_tests = sorted(list(set([ex.timestep for ex in testset])))
    test_loaders = []
    if opt.config == 'prediction':
        test_loaders.append((ts_tests[0] - 1, val_loader))
    for t in ts_tests:
        test_t = Dataset(testset.examples, testset.fields, filter_pred=lambda x: x.timestep == t)
        test_t.sort_key = lambda x: len(x.text)
        test_t_loader = BucketIterator(test_t, opt.batch_size, train=False, device=device)
        test_loaders.append((t, test_t_loader))
    test_loaders = OrderedDict(test_loaders)
    # opt
    opt.ntoken = corpus.vocab_size
    opt.padding_idx = corpus.pad_idx
    opt.nts_train = max(ex.timestep for ex in trainset) + 1
    opt.nwords_train = sum(len(ex.text) for ex in trainset)

    ##################################################################################################################
    # Model
    ##################################################################################################################
    # load config
    model_opt = load_config(os.path.join('configs', opt.corpus, opt.config, '{}.yaml'.format(opt.model)))
    opt = DotDict({**opt, **model_opt})
    # buid model
    print('Building model...')
    model = lm_factory(opt).to(device)

    ##################################################################################################################
    # Optimizer
    ##################################################################################################################
    optimizer = torch.optim.Adam(get_lm_parameters(model, opt), lr=opt.lr, eps=1e-9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience, factor=opt.lr_decay)

    ##################################################################################################################
    # Log
    ##################################################################################################################
    opt.xproot = os.path.join(opt.xproot, opt.corpus, opt.config, opt.model, opt.name)
    if not os.path.isdir(opt.xproot):
        os.makedirs(opt.xproot)
    print('Experiment directory: {}'.format(opt.xproot))
    with open(os.path.join(opt.xproot, 'config.json'), 'w') as f:
        json.dump(opt, f, sort_keys=True, indent=4)

    ##################################################################################################################
    # Trainning
    ##################################################################################################################
    print('Training...')
    pb = trange(opt.nepoch, ncols=0)
    try:
        for e in pb:
            model.train()
            for batch in train_loader:
                # inputs
                text = batch.text[0][:-1]
                target = batch.text[0][1:]
                timestep = batch.timestep
                # zero grad
                optimizer.zero_grad()
                # closure
                loss = model.closure(text, target, timestep)
                # backward
                loss.backward()
                # step
                optimizer.step()
            model.eval()
            with torch.no_grad():
                _, ppl_eval, _ = evaluate_lm(model, val_loader, opt)
            lr_scheduler.step(ppl_eval)
            lr = optimizer.param_groups[0]['lr']
            if lr < 1e-7:
                break
            pb.set_postfix(loss=loss.item(), ppl_eval=ppl_eval, lr=lr)
    except KeyboardInterrupt:
        pass
    pb.close()
    print('Evaluating...')
    ntkn_test = 0
    nlls = []
    ppls = []
    model.eval()
    with torch.no_grad():
        for t, loader in test_loaders.items():
            nll, ppl, ntkn = evaluate_lm(model, loader, opt)
            nlls.append(nll)
            ppls.append((t, ppl))
            ntkn_test += ntkn
    results = [
        ('epoch', e),
        ('micro', perplexity(sum(nlls) / ntkn_test)),
        ('macro', sum(perplexity(nll) for nll in nlls) / len(nlls)),
    ]
    results = OrderedDict(results + ppls)
    print('Saving results...')
    with open(os.path.join(opt.xproot, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    torch.save(model.state_dict(), os.path.join(opt.xproot, 'model.pt'))
    print('Done')


if __name__ == '__main__':
    # arguments
    p = configargparse.ArgParser()
    p.add('--corpus', required=True, type=str, help='Corpus name')
    p.add('--config', required=True, type=str, help='Evaluation configuration: prediction | modeling')
    p.add('--model', required=True, type=str, help='Model name: lstm | drlm')
    p.add('--xproot', type=str, default='/local/delasalles/xp/drlm', help='Base saving directory')
    p.add('--name', type=str, default='xp', help='Name of the experiment')
    p.add('--batch_size', type=int, default=64)
    p.add('--nepoch', type=int, default=1000)
    p.add('--device', type=int, default=-1, help='-1: cpu; > -1: cuda device id')
    p.add('--manualSeed', type=int, help='manual seed')
    # parse
    opt = p.parse_args()
    # main
    main(DotDict(vars(opt)))
