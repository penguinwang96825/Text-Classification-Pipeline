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

    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            self.PAD_token: "[PAD]", self.UNK_token: "[UNK]", self.BOS_token: "[BOS]", self.EOS_token: "[EOS]"
        }
        self.word2index = {word:index for index, word in self.index2word.items()}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0

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
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in tokenise(sentence):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def index_to_word(self, index):
        return self.index2word[index]

    def word_to_index(self, word):
        return self.word2index.get(word, self.UNK_token)