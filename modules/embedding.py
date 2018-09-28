import torch.nn as nn


class Embedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, dropoute=.0, dropout=.0, **kwargs):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, **kwargs)
        self.dropoute = dropoute
        self.drop = nn.Dropout(dropout)
        self.weight.data.uniform_(-0.1, 0.1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx] = 0

    def forward(self, input):
        emb = super(Embedding, self).forward(input)
        if self.training and self.dropoute > 0:
            input_flatten = input.flatten()
            mask = emb.new(self.num_embeddings).bernoulli_(1 - self.dropoute) / (1 - self.dropoute)
            mask = mask[input_flatten].view_as(input).unsqueeze(-1).expand_as(emb)
            emb = emb * mask
        return self.drop(emb)
