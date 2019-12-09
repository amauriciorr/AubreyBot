import re
import pickle as pkl
from collections import defaultdict


RETOK = re.compile(r'\w+|[^\w\s\[\]]|\n', re.UNICODE)
SONG_PART_REGEX = re.compile(r'\[\w+\W?\s?\w*\d?\]', re.UNICODE)

def create_counts_dict(lyrics_file, regex):
    word_count_dict = defaultdict(int)
    for song in lyrics_file:
        for word in regex.findall(song['lyrics']):
            word_count_dict[word] += 1
    return word_count_dict

def get_all_lyrics(lyrics_file, regex):
    all_lyrics = []
    for song in lyrics_file:
        lyrics = regex.sub('', song['lyrics'])
        lyrics = lyrics.lower()
        all_lyrics.append(lyrics)
    return all_lyrics

def create_text_and_target():
    """
    TODO
    growing window that adds subsequent line as target
    assembles text and target in JSON file
    """
    pass



class ChatDictionary(object):
    """
    Simple dict loader
    """
    def __init__(self, dict_file_path):
        self.word2ind = {}  # word:index
        self.ind2word = {}  # index:word
        self.counts = {}  # word:count

        dict_raw = open(dict_file_path, 'r').readlines()
        
        for i, w in enumerate(dict_raw):
            _word, _count = w.strip().split('\t')
            if _word == '\\n':
                _word = '\n'
            self.word2ind[_word] = i
            self.ind2word[i] = _word
            self.counts[_word] = _count
            
    def t2v(self, tokenized_text):
        return [self.word2ind[w] if w in self.counts else self.word2ind['__unk__'] for w in tokenized_text]

    def v2t(self, list_ids):
        return ' '.join([self.ind2word[i] for i in list_ids])
    
    def pred2text(self, tensor):
        result = []
        for i in range(tensor.size(0)):
            if tensor[i].item() == '__end__'  or tensor[i].item() == '__null__':  # null is pad
                break
            else:
                result.append(self.ind2word[tensor[i].item()])
        return ' '.join(result)
    
    def __len__(self):
        return len(self.counts)

