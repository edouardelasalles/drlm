import configargparse
import os
import json
import random
from functools import partial
from collections import defaultdict, OrderedDict, Counter

from tqdm import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
from torchtext.data import Iterator, Dataset

from corpus import Corpus
from evaluate import evaluate_lm, evaluate_lm_at_t
from utils import DotDict, Logger, load_config, lm_factory, get_lm_optimizer


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
    print(f'{len(trainset)} training documents with {opt.nwords} tokens on {opt.nts} timesteps')

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
    optimizer = get_lm_optimizer(model, opt)
    if 'lr_scheduling' in opt:
        if opt.lr_scheduling == 'linear':
            opt.min_lr == 0
            opt.niter = opt.niter_burnin + opt.niter_scheduling
            niter = opt.niter_scheduling
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                             lr_lambda=lambda i: max(0, (niter - i) / niter))
        if opt.lr_scheduling == 'reduce_on_plateau':
            assert opt.min_lr > 0
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      patience=opt.patience, factor=opt.lr_decay)
    else:
        lr_scheduler = None

    ##################################################################################################################
    # Log
    ##################################################################################################################
    opt.xproot = os.path.join(opt.xproot, opt.corpus, opt.config, opt.model, opt.name)
    print(f'New experiment logged at {opt.xproot}')
    logger = Logger(opt.xproot)
    logger.init(opt)

    ##################################################################################################################
    # Trainning
    ##################################################################################################################
    print('Training...')
    pb = trange(opt.niter, ncols=0)
    ppl_eval = None
    finished = False
    itr = -1
    try:
        while not finished:
            for batch in train_loader:
                itr += 1
                model.train()
                # io
                text = batch.text[0][:-1]
                target = batch.text[0][1:]
                timestep = batch.timestep
                # closure
                log_train = model.closure(text, target, timestep, optimizer, opt)
                # eval
                if itr > 0 and itr % opt.niter_checkpoint == 0:
                    model.eval()
                    with torch.no_grad():
                        score, log_val = evaluate_lm(model, val_loaders, opt)
                    # checkpoint
                    log_train['lr'] = optimizer.param_groups[0]['lr']
                    logger.log(itr, 'train', log_train)
                    logger.log(itr, 'val', log_val)
                    logger.checkpoint(itr)
                    # reduce_on_plateau lr scheduling
                    if lr_scheduler and itr >= opt.niter_burnin and opt.lr_scheduling == 'reduce_on_plateau':
                        lr_scheduler.step(score)
                    lr = optimizer.param_groups[0]['lr']
                    if lr < opt.min_lr:
                        finished = True
                        break
                    # progress bar
                    pb.update(opt.niter_checkpoint)
                    pb.set_postfix(chkpt=logger.chkpt, loss=log_train['loss'], score=score, lr=lr)
                # other lr scheduling
                if lr_scheduler and itr >= opt.niter_burnin and opt.lr_scheduling != 'reduce_on_plateau':
                    lr_scheduler.step()
                lr = optimizer.param_groups[0]['lr']
                if lr < opt.min_lr:
                    finished = True
    except KeyboardInterrupt:
        exit_code = 130
    pb.close()
    print('Evaluating...')
    model.eval()
    with torch.no_grad():
        _, log_val = evaluate_lm(model, val_loaders, opt)
        _, results = evaluate_lm(model, test_loaders, opt)
    log_train['lr'] = optimizer.param_groups[0]['lr']
    logger.log(itr, 'train', log_train)
    logger.log(itr, 'val', log_val)
    logger.log(itr, 'test', results)
    logger.checkpoint(itr)
    logger.terminate(model, optimizer)
    return exit_code


if __name__ == '__main__':
    # arguments
    p = configargparse.ArgParser()
    p.add('--xproot', type=str, default='xp', help='Base saving directory')
    p.add('--corpus', required=True, type=str, help='Corpus name')
    p.add('--config', required=True, type=str, help='Evaluation configuration: prediction | modeling')
    p.add('--model', required=True, type=str, help='Model name: lstm | drlm')
    p.add('--name', type=str, default='debug', help='Experiment name')
    p.add('--device', type=int, default=-1, help='-1: cpu; > -1: cuda device id')
    p.add('--manualSeed', type=int, help='manual seed')
    # parse
    opt = p.parse_args()
    # main
    main(DotDict(vars(opt)))
