import os
import argparse
import logging
import torch.optim as optim
import torch.nn.functional as F
from utils.utils import load_dataset
from base.base_dataloader import BaseDataLoader
from model.nnets import LogRes
os.system('rm -f -r */__pycache__')
logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config):
    logger.info('Loading dataset...')
    dataset = load_dataset(config.dataset, config.maxlen)
    train_dataloader = BaseDataLoader(dataset, config.batch_size, shuffle=True, validation_split=config.validation_split, num_workers=0)
    valid_dataloader = train_dataloader.split_validation()

    logger.info('Modelling...')
    vocab_size = len(dataset.vocab.index2word)
    model = LogRes(vocab_size=vocab_size, embedding_dim=300, num_classes=2, dropout=0.1)
    model.compile(F.cross_entropy, optim.Adam(model.parameters()))
    model.fit(train_dataloader, valid_dataloader, config.max_epoch, gpu=True)
    model.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PyTorch model...')
    parser.add_argument('-d', '--dataset', default=None, type=str,
                        help='')
    parser.add_argument('-l', '--maxlen', default=None, type=int,
                        help='')
    parser.add_argument('-b', '--batch_size', default=None, type=int,
                        help='')
    parser.add_argument('-v', '--validation_split', default=None, type=float,
                        help='')
    parser.add_argument('-e', '--max_epoch', default=None, type=int,
                        help='')
    args = parser.parse_args()
    
    main(args)