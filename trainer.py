import configargparse
import os
import json
import random
from collections import defaultdict, OrderedDict, Counter

from tqdm import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
from torchtext.data import Iterator, Dataset

from corpus import Corpus
from utils import DotDict, load_config, lm_factory, evaluate_lm, evaluate_lm_at_t, get_lm_optimizers, get_lr


def main(opt):
    exit_code = 0
    opt.hostname = os.uname()[1]
    opt.running = True
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
    data_opt = load_config(os.path.join('config', opt.corpus, opt.config, 'corpus.yaml'))
    opt.update(data_opt)
    # load data
    corpus = Corpus(opt.dataroot)
    # split
    trainset, valset, testset = corpus.split(opt.config, opt.min_freq)
    # dataloaders
    # -- train
    train_loader = Iterator(trainset, opt.batch_size, repeat=False, sort_within_batch=True, device=device)
    # -- val
    ts_val = sorted(list(set([ex.timestep for ex in valset])))
    val_loaders = []
    for t in ts_val:
        val_t = Dataset(valset.examples, valset.fields, filter_pred=lambda x: x.timestep == t)
        val_t.sort_key = lambda x: len(x.text)
        val_t_loader = Iterator(val_t, opt.batch_size, train=False, device=device)
        val_loaders.append((t, val_t_loader))
    val_loaders = OrderedDict(val_loaders)
    # -- test
    ts_tests = sorted(list(set([ex.timestep for ex in testset])))
    test_loaders = []
    if opt.config == 'prediction':
        for t, loader in val_loaders.items():
            test_loaders.append((t, loader))
    for t in ts_tests:
        test_t = Dataset(testset.examples, testset.fields, filter_pred=lambda x: x.timestep == t)
        test_t.sort_key = lambda x: len(x.text)
        test_t_loader = Iterator(test_t, opt.batch_size, train=False, device=device)
        test_loaders.append((t, test_t_loader))
    test_loaders = OrderedDict(test_loaders)
    # opt
    opt.ntoken = corpus.vocab_size
    opt.padding_idx = corpus.pad_idx
    opt.nts = max(ex.timestep for ex in trainset) + 1
    opt.nwords = sum(len(ex.text) for ex in trainset)
    # print info
    print('Vocab size: {}'.format(opt.ntoken))
    print('Number of training documents: {}'.format(len(trainset)))
    print('Number of training tokens: {}'.format(opt.nwords))

    ##################################################################################################################
    # Model
    ##################################################################################################################
    # load config
    model_opt = load_config(os.path.join('config', opt.corpus, opt.config, '{}.yaml'.format(opt.model)))
    opt.update(model_opt)
    # buid model
    print('Building model...')
    model = lm_factory(opt).to(device)

    ##################################################################################################################
    # Optimizer
    ##################################################################################################################
    optimizers = get_lm_optimizers(model, opt)
    lr_schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience, factor=opt.lr_decay)
                     for optimizer in optimizers.values()]

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
                # closure
                loss = model.closure(text, target, timestep, optimizers, opt)
            # eval
            model.eval()
            with torch.no_grad():
                eval_ppls = evaluate_lm(model, val_loaders, opt)
                ppl_eval = eval_ppls['micro']
            # schedule lr
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step(ppl_eval)
            lr = get_lr(optimizers, opt)
            if lr < 1e-6:
                break
            # progress bar
            pb.set_postfix(loss=loss, ppl_eval=ppl_eval, lr=lr)
    except KeyboardInterrupt:
        exit_code = 130
    pb.close()
    print('Evaluating...')
    results = OrderedDict([('epoch', e)])
    results.update(evaluate_lm(model, test_loaders, opt))
    print('Saving results...')
    with open(os.path.join(opt.xproot, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    torch.save(model.state_dict(), os.path.join(opt.xproot, 'model.pt'))
    opt.running = False
    with open(os.path.join(opt.xproot, 'config.json'), 'w') as f:
        json.dump(opt, f, sort_keys=True, indent=4)
    print('Done')
    return exit_code


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
