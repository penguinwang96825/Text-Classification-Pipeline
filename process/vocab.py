import numpy as np
from copy import deepcopy
from gensim.parsing.preprocessing import STOPWORDS
from .preprocess import tokenise


class Vocabulary(object):
    PAD_token = 0   # Used for padding short sentences
    UNK_token = 1   # Unknown word
    BOS_token = 2   # Begin-of-sentence token
    EOS_token = 3   # End-of-sentence token
    
    """
    Examples
    --------
    >>> corpus = [
    ...     'Acting CoC Hsu More crypto regulation is needed', 
    ...     'Argo Blockchain's Texas mining facility could cost up to $2B', 
    ...     'New study reveals which US cities lead crypto hires in 2021'
    ... ]
    >>> vocab = Vocabulary()
    >>> for doc in corpus:
    ...     vocab.add_sentence(doc)
    """

    def __init__(self, stop_words=None, max_features=None):
        self.stop_words = stop_words
        self.max_features = max_features

        self.special_tokens = {
            "[PAD]":self.PAD_token, "[UNK]":self.UNK_token, "[BOS]":self.BOS_token, "[EOS]":self.EOS_token
        }
        self.word2index = {}
        self.word2count = {
            "[PAD]":np.inf, "[UNK]":np.inf, "[BOS]":np.inf, "[EOS]":np.inf
        }
        self.word2index = deepcopy(self.special_tokens)
        self.index2word = {index:word for word, index in self.word2index.items()}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0

        self._build_stop_words()

    def __str__(self):
        return f"<Vocabulary(num_vocabs={len(self.num_words)})>"

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def _get_top_features(self, word2count):
        return dict(sorted(word2count.items(), key=lambda x: x[1], reverse=True)[:self.max_features])

    def _sort_wordmap(self, wordmap):
        return dict(sorted(wordmap.items(), key=lambda x: x[1], reverse=False))

    def _reset_wordmap(self, wordmap):
        return {word:i for i, (word, index) in enumerate(wordmap.items())}

    def add_sentence(self, sentence, tokenise_fn=None):
        if tokenise_fn is None:
            tokenise_fn = tokenise
        sentence_len = 0
        for word in tokenise_fn(sentence):
            if word not in self.stop_words:
                sentence_len += 1
                self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

        # Consider the top max_features ordered by term frequency across the corpus
        if self.max_features is None:
            self.max_features = self.num_words
        self.word2count = self._get_top_features(self.word2count)
        self.word2index = {word:index for word, index in self.word2index.items() if word in self.word2count.keys()}
        self.word2index = {**self.word2index, **self.special_tokens}
        self.word2index = self._sort_wordmap(self.word2index)
        self.word2index = self._reset_wordmap(self.word2index)

    def build(self, corpus):
        for doc in corpus:
            self.add_sentence(doc)

    def _build_stop_words(self):
        if self.stop_words is None:
            self.stop_words = []
        elif self.stop_words == 'english':
            self.stop_words = STOPWORDS

    def index_to_word(self, index):
        return self.index2word[index]

    def word_to_index(self, word):
        return self.word2index.get(word, self.UNK_token)


def encode(text, vocab: Vocabulary, tokenise_fn=None, stop_words=STOPWORDS):
    if tokenise_fn is None:
        tokenise_fn = tokenise
    tokens = tokenise_fn(text)
    ids = [vocab.word_to_index(token) for token in tokens if token not in stop_words]
    return ids