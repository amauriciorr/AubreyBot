import re
import pickle as pkl
from collections import defaultdict


RETOK = re.compile(r'\w+|[^\w\s\[\]]|\n', re.UNICODE)
SONG_PART_REGEX = re.compile(r'\[\w+\W?\s?\w*\d?\]', re.UNICODE)

def create_counts_dict(lyrics_file, regex):
    '''
    function for determining frequency of occurrence 
    for each word
    '''
    word_count_dict = defaultdict(int)
    for song in lyrics_file['songs']:
        for word in regex.findall(song['lyrics']):
            word_count_dict[word.lower()] += 1
    return word_count_dict

def get_all_lyrics(lyrics_file, regex):
    '''
    iterate through Genius JSON file and retrieve
    lyrics. remove song-part designation
    '''
    all_lyrics = []
    for song in lyrics_file['songs']:
        lyrics = regex.sub('', song['lyrics'])
        lyrics = lyrics.lower()
        all_lyrics.append(lyrics)
    return all_lyrics

def create_text_and_target(song):

    text_and_target_dict = {}
    song = song.split('\n')
    # 0-index refers to song-part designation that was replaced
    # by empty string, i.e. [INTRO]
    base_lyric = song[1]
    for lyric in song[2:]:
        if lyric:
            text_and_target_dict[base_lyric] = lyric
            base_lyric += ('\n ' + lyric)

    '''
    TO DO 
    LOOK AT PERSONACHAT JSON FILE
    MAKE SAME KEY-VALUE PAIR STRUCTURE HERE
    '''

    return text_and_target_dict

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


class ChatDataset(Dataset):
    """
    Json dataset wrapper
    """
    
    def __init__(self, dataset_file_path, dictionary):
        super().__init__()
        
        json_text = open(dataset_file_path, 'r').readlines()
        self.samples = []
        
        for sample in tqdm(json_text):
            sample = sample.rstrip()
            sample = json.loads(sample)
            _inp_toked = RETOK.findall(sample['text'])
            _inp_toked_id = dictionary.t2v(_inp_toked)

            sample['text_vec'] = torch.tensor(_inp_toked_id, dtype=torch.long)

            _tar_toked = RETOK.findall(sample['labels'][0]) + ['__end__']
            _tar_toked_id = dictionary.t2v(_tar_toked)
            
            sample['target_vec'] = torch.tensor(_tar_toked_id, dtype=torch.long)
            
            self.samples.append(sample)
            
    def __getitem__(self, i):
        return self.samples[i]['text_vec'], self.samples[i]['target_vec']
    
    def __len__(self):
        return len(self.samples)
