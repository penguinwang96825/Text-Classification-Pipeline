import torch
import torch.nn as nn


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_model import BaseModel
from base.base_trainer import Trainer


class LogRes(BaseModel, Trainer):

    def __init__(self, vocab_size=20000, embedding_dim=300, num_classes=2, dropout=0.1):
        super(LogRes, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(self.drop(x))
        return x