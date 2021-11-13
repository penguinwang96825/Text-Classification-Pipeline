import pandas as pd
from torch.utils.data import Dataset


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from process.preprocess import tokenise, decontracted, pad_sequences
from process.vocab import Vocabulary


class IMDB(Dataset):
    PAD_token = 0   # Used for padding short sentences
    UNK_token = 1   # Unknown word
    BOS_token = 2   # Begin-of-sentence token
    EOS_token = 3   # End-of-sentence token

    def __init__(self, do_lower_case=True, maxlen=64, add_special_tokens=True):
        super(IMDB, self).__init__()
        text_df = download_imdb()
        label_map = {'negative':0, 'positive':1}
        text_df['review'] = text_df['review'].apply(self._preprocess)
        self.vocab = Vocabulary()
        for doc in text_df['review']:
            self.vocab.add_sentence(doc)
        
        self.reviews = text_df['review'].tolist()
        self.tokens = text_df['review'].apply(
            lambda x: ['[BOS]'] + tokenise(x, lower=do_lower_case) + ['[EOS]'] if add_special_tokens else tokenise(x, lower=do_lower_case)
        ).tolist()
        self.ids = list(map(self._to_sequence, self.tokens))
        self.ids = pad_sequences(self.ids, maxlen=maxlen, value=0, truncating='post', padding='post')
        self.labels = text_df['sentiment'].map(label_map).tolist()

    def _preprocess(self, text: str):
        text = decontracted(text)
        return text

    def _to_sequence(self, tokens: list):
        return [self.vocab.word_to_index(token) for token in tokens]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        ids = self.ids[index]
        label = self.labels[index]
        return ids, label


def download_imdb():
    url = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
    df = pd.read_csv(url)
    return df
