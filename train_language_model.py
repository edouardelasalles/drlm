import configargparse
import os
import json
import random
from collections import defaultdict, OrderedDict

from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchtext.data import Iterator, BucketIterator, Dataset

from corpus import Corpus
from drlm import DynamicRecurrentLanguageModel
from utils import DotDict, perplexity


def evaluate(model, dataloader, opt):
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
    # load data
    corpus = Corpus(opt.dataroot)
    # split
    trainset, valset, testset = corpus.split(opt.mode, opt.min_freq)
    # dataloader
    train_loader = Iterator(trainset, opt.batch_size, repeat=False, sort_within_batch=True, device=device)
    val_loader = BucketIterator(valset, opt.batch_size, train=False, device=device)
    ts_tests = sorted(list(set([ex.timestep for ex in testset])))
    test_loaders = []
    if opt.mode == 'prediction':
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
    # log opt
    opt.xproot = os.path.join(opt.xproot, opt.name)
    if not os.path.isdir(opt.xproot):
        os.makedirs(opt.xproot)
    print('Experiment directory: {}'.format(opt.xproot))
    with open(os.path.join(opt.xproot, 'config.json'), 'w') as f:
        json.dump(opt, f, sort_keys=True, indent=4)

    ##################################################################################################################
    # Model
    ##################################################################################################################
    print('Building model...')
    model = DynamicRecurrentLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute,
                                          opt.dropouti, opt.dropoutl, opt.dropouth, opt.dropouto, opt.tie_weights,
                                          opt.nts_train, opt.nzt, opt.nhid_zt, opt.nlayers_zt, opt.res_zt,
                                          not opt.no_transition, opt.padding_idx, opt.nwords_train).to(device)

    ##################################################################################################################
    # Optimizer
    ##################################################################################################################
    optimizer = torch.optim.Adam(model.get_parameters(opt.wd, opt.wd_t), lr=opt.lr, eps=1e-9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience, factor=0.1)

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
                _, ppl_eval, _ = evaluate(model, val_loader, opt)
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
            nll, ppl, ntkn = evaluate(model, loader, opt)
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
    p = configargparse.ArgParser()
    p.add('--config', is_config_file=True, help='Config file path')
    # -- data
    p.add('--dataroot', required=True, type=str, help='Directory containing a dataset')
    p.add('--mode', required=True, type=str, default='prediction', help='Evaluation configuration : prediction | modeling')
    p.add('--min_freq', required=True, type=int, help='min word frequency')
    # -- xp
    p.add('--xproot', required=True, type=str, help='Directory where models will be saved')
    p.add('--name', type=str, required=True, help='Name of the experiment')
    # -- word embeddings
    p.add('--nwe', type=int, default=400, help='Word embeddings size')
    # -- language model
    p.add('--nhid_rnn', type=int, default=400, help='LSTM hidden size')
    p.add('--nlayers_rnn', type=int, default=2, help='Number of layers in LSTM')
    p.add('--tie_weights', action='store_true', help='enable weight sharing between word embeddings and decoder')
    p.add('--dropoute', type=float, required=True, help='Word embeddings dropout')
    p.add('--dropouti', type=float, required=True, help='Input dropout')
    p.add('--dropoutl', type=float, required=True, help='Inter layers dropout')
    p.add('--dropouth', type=float, required=True, help='Weight dropout')
    p.add('--dropouto', type=float, required=True, help='Output dropout')
    # -- DRLM parameters
    p.add('--nzt', type=int, required=True, help='Latent sate size')
    p.add('--no_transition', action='store_true', help='Do not learn transition function ?')
    p.add('--nhid_zt', type=int, required=True, help='Tansition function hidden size')
    p.add('--nlayers_zt', type=int, required=True, help='Number of layers for transition function')
    p.add('--res_zt', action='store_true', help='Residual transition function ?')
    p.add('--wd_t', type=float, required=True, help='Transition function weight decay')
    # -- optim
    p.add('--lr', type=float, required=True, help='Learning rate')
    p.add('--wd', type=float, required=True, help='LSTM weight decay')
    p.add('--lr_decay', type=float, default=0.1, help='learning rate decay')
    p.add('--patience', type=int, required=True)
    # -- learning
    p.add('--batch_size', type=int, default=64)
    p.add('--nepoch', type=int, default=1000)
    # -- gpu
    p.add('--device', type=int, default=-1, help='-1: cpu; > -1: cuda device id')
    # -- seed
    p.add('--manualSeed', type=int, help='manual seed')
    # parse
    opt = p.parse_args()
    # main
    main(DotDict(vars(opt)))
