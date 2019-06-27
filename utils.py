import os
import shutil
import subprocess
import json
import yaml
from os.path import join
from numbers import Number
from collections import OrderedDict

import torch

from model.lstm import LSTMLanguageModel
from model.drlm import DynamicRecurrentLanguageModel


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path):
    with open(path, 'r') as f:
        return DotDict(yaml.safe_load(f))


def lm_factory(opt):
    if opt.model in ('drlm', 'drlm_id'):
        return DynamicRecurrentLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute,
                                             opt.dropouti, opt.dropoutl, opt.dropouth, opt.dropouto, opt.tied_weights,
                                             opt.nts, opt.nzt, opt.nhid_zt, opt.nlayers_zt, opt.learn_transition,
                                             opt.padding_idx, opt.nwords)
    if opt.model == 'lstm':
        return LSTMLanguageModel(opt.ntoken, opt.nwe, opt.nhid_rnn, opt.nlayers_rnn, opt.dropoute, opt.dropouti,
                                 opt.dropoutl, opt.dropouth, opt.dropouto, opt.tied_weights, opt.padding_idx,
                                 opt.nwords)
    raise ValueError('No model named `{}`'.format(opt.model))


def get_lm_optimizer(model, opt):
    return model.get_optimizer(opt.lr, opt.wd_lm)


def save_pt(pt_object, path):
    try:
        torch.save(pt_object, path)
        success = True
    except Exception as e:
        print('Warning: could not save pytorch object at {}'.format(path))
        print(e)
        success = False
    return success


def save_json(data_dict, path):
    ordered = isinstance(data_dict, OrderedDict)
    try:
        with open(path, 'w') as f:
            json.dump(data_dict, f, sort_keys=not ordered, indent=4)
        success = True
    except Exception as e:
        print('Warning: could not save json object at {}'.format(path))
        print(e)
        success = False
    return success


def save_src(path):
    current_dir = os.getcwd()
    src_files = subprocess.Popen(('find', '.',
                                  '-name', '*.py',
                                  '-o', '-name', '*.yaml',
                                  '-o', '-name', '*.json'),
                                 stdout=subprocess.PIPE)
    subprocess.check_output(('tar', '-zcf', path, '-T', '-'), stdin=src_files.stdout)
    src_files.wait()
    os.chdir(current_dir)


class Logger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if os.path.isdir(log_dir):
            remove = input("Experiment directory already exists. Remove? (y|n)")
            if remove == 'y':
                shutil.rmtree(log_dir)
            else:
                print('Aborting experiment...')
                exit()
        os.makedirs(log_dir)
        self.logs = OrderedDict()
        self.chkpt = 0
        self.save2disk_freq = 20

    def init(self, opt):
        # save xp opt
        opt['running'] = True
        save_json(opt, join(self.log_dir, 'config.json'))
        save_src(join(self.log_dir, 'src.tar.gz'))

    def log(self, itr, keys, val, log_dict=None):
        log_dict = self.logs if log_dict is None else log_dict
        key = keys if isinstance(keys, str) else keys[-1]
        keys = [key] if isinstance(keys, str) else keys
        if isinstance(val, dict):
            if key not in log_dict:
                log_dict[key] = OrderedDict()
            for k, v in val.items():
                self.log(itr, keys + [k], v, log_dict=log_dict[key])
        else:
            if key not in log_dict:
                log_dict[key] = []
            if isinstance(val, Number):
                log_dict[key].append(val)
            elif isinstance(val, list):
                for v in val:
                    assert isinstance(v, Number), f'trying to log {key}, but it is not a number'
                log_dict[key].append(val)
            else:
                raise TypeError(f'Failed to log `{key}`. Logging `{type(val)}` is not supported')

    def _dump(self, model=None, optimizer=None):
        success = True
        # write logs on json file
        for k, v in self.logs.items():
            success = success and save_json(v, join(self.log_dir, 'logs.{}.json'.format(k)))
        if model is not None:
            success = success and save_pt(model.state_dict(), join(self.log_dir, 'model.pt'))
        if optimizer is not None:
            success = success and save_pt(model.state_dict(), join(self.log_dir, 'model.pt'))
        return success

    def checkpoint(self, i):
        self.chkpt += 1
        if self.chkpt % self.save2disk_freq == 0:
            self._dump()

    def terminate(self, model, optimizer):
        self.chkpt += 1
        success = False
        i = 0
        pause = 300
        while not success:
            i += 1
            if i > 1:
                if i <= self.n_try:
                    print('Failed to save final checkpoint, retrying...')
                    time.sleep(pause)
                    pause *= 2
                else:
                    input('Failed to save xp. Press any key to retry')
            try:
                config_path = join(self.log_dir, 'config.json')
                with open(config_path, 'r') as info:
                    opt = json.load(info)
            except:
                continue
            opt['running'] = False
            success = save_json(opt, config_path)
            if not success:
                continue
            # dump stack
            success = self._dump(model, optimizer)
