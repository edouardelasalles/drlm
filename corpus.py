import os
import json
from collections import OrderedDict

import torch
from torchtext.data import Dataset, Field, Example
from torchtext.vocab import Vocab


class Corpus(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        fname = 'corpus.json'
        # fields
        id_field = Field(sequential=False, unk_token=None)
        text_field = Field(include_lengths=True)
        timestep_field = Field(sequential=False, use_vocab=False, unk_token=None)
        fields = [('id', id_field), ('text', text_field), ('timestep', timestep_field)]
        # load examples
        fpath = os.path.join(data_dir, 'corpus.json')
        print('Loading {}...'.format(fpath))
        with open(fpath, 'r') as f:
            corpus = json.load(f)
        examples = [Example.fromlist([ex['id'], ex['text'], ex['timestep']], fields) for ex in corpus]
        dataset = Dataset(examples, fields)
        id_field.build_vocab(dataset)
        self.examples = examples
        self.fields = OrderedDict(fields)
        self.nts = max([ex.timestep for ex in self.examples]) + 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def load_dataset(self, mode, fold):
        with open(os.path.join(self.data_dir, '{}_{}.txt'.format(mode, fold))) as f:
            ids = f.readlines()
        ids = set([idx.strip() for idx in ids])
        dset = Dataset(self.examples, self.fields, filter_pred=lambda x: x.id in ids)
        dset.sort_key = lambda x: len(x.text)
        return dset

    def split(self, mode, min_freq):
        if mode == 'prediction':
            mode = 'pred'
        elif mode == 'modeling':
            mode = 'model'
        else:
            raise ValueError('`mode` parameter should be `prediction` or `modeling`, got `{}`'.format(mode))
        # folds
        trainset = self.load_dataset(mode, 'train')
        valset = self.load_dataset(mode, 'val')
        testset = self.load_dataset(mode, 'test')
        # vocab
        self.fields['text'].eos_token = '<eos>'
        self.fields['text'].build_vocab(trainset, min_freq=min_freq)
        self.fields['text'].eos_token = None
        self.vocab = self.fields['text'].vocab
        self.vocab_size = len(self.vocab.itos)
        self.pad_idx = self.vocab.stoi['<pad>']
        return trainset, valset, testset
