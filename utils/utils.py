import os
import torch
import random
import numpy as np


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def seed_everything(seed=914):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset, maxlen):
    if dataset == 'imdb':
        from dataset.imdb import IMDB
        dataset = IMDB(maxlen=maxlen)
        return dataset