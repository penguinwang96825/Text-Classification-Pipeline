import re
import numpy as np
from textblob import TextBlob


def tokenise(text, errors="strict", lower=False):

    def to_unicode(text, encoding='utf8', errors='strict'):
        if isinstance(text, str):
            return text
        return str(text, encoding, errors=errors)

    def _tokenise(text, errors, lower):
        PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)
        text = to_unicode(text, errors=errors)
        if lower:
            text = text.lower()
        for match in PAT_ALPHABETIC.finditer(text):
            yield match.group()

    return list(_tokenise(text, errors, lower))


def decontracted(text):
    # Specific case
    text = re.sub(r"won(\'|\’)t", "will not", text)
    text = re.sub(r"can(\'|\’)t", "can not", text)

    # General case
    text = re.sub(r"n(\'|\’)t", " not", text)
    text = re.sub(r"(\'|\’)re", " are", text)
    text = re.sub(r"(\'|\’)s", " is", text)
    text = re.sub(r"(\'|\’)d", " would", text)
    text = re.sub(r"(\'|\’)ll", " will", text)
    text = re.sub(r"(\'|\’)t", " not", text)
    text = re.sub(r"(\'|\’)ve", " have", text)
    text = re.sub(r"(\'|\’)m", " am", text)
    return text


def lemmatize_with_postag(text):
    sent = TextBlob(text)
    tag_dict = {
        "J": 'a', 
        "N": 'n', 
        "V": 'v', 
        "R": 'r'
    }
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatised_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatised_list)


def pad_sequences(sequences, maxlen=64, value=0, truncating='pre', padding='post'):
    if padding == 'post':
        if truncating == 'post':
            pad_ = lambda seq, maxlen : seq[0:maxlen] if len(seq) > maxlen else seq + [value] * (maxlen-len(seq))
        elif truncating == 'pre':
            pad_ = lambda seq, maxlen : seq[-maxlen:] if len(seq) > maxlen else seq + [value] * (maxlen-len(seq))
    elif padding == 'pre':
        if truncating == 'post':
            pad_ = lambda seq, maxlen : seq[0:maxlen] if len(seq) > maxlen else [value] * (maxlen-len(seq)) + seq
        elif truncating == 'pre':
            pad_ = lambda seq, maxlen : seq[-maxlen:] if len(seq) > maxlen else [value] * (maxlen-len(seq)) + seq
    padded_sequences = [pad_(seq, maxlen) for seq in sequences]
    return np.stack(padded_sequences)