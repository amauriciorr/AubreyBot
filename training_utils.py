import re
import json
import torch
import numpy as np
import datetime as dt
import pickle as pkl
import pytz
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from processing_utils import RETOK
# commented above, added below to run on GCP without needing to import
# processing_utils, i.e. installing lyrics_genius library
RETOK = re.compile(r'\w+|[^\w\s\[\]]|\n', re.UNICODE)
EPOCH_LOG = '{} | Epoch {} / {}'
STEP_LOG = '{} | Perplexity achieved: {} | Step {} / {}'
TIMEZONE = pytz.timezone('America/New_York')

class ChatDictionary(object):
    """
    Simple dict loader
    """
    def __init__(self, dict_file_path):
        self.word2ind = {}  # word:index
        self.ind2word = {}  # index:word
        self.counts = {}  # word:count
        # dict_raw = open(dict_file_path, 'r').readlines()
        dict_raw = pkl.load(open(dict_file_path, 'rb'))

        for i, kvp in enumerate(dict_raw.items()):
            _word, _count = kvp
            self.word2ind[_word] = i
            self.ind2word[i] = _word
            self.counts[_word] = _count

    def t2v(self, tokenized_text):
        return [self.word2ind[w] if w in self.counts else self.word2ind['__unk__']
                for w in tokenized_text]

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
    def __init__(self, dictionary, dataset_file_path, dt='train'):
        super().__init__()

        json_text = open(dataset_file_path, 'r').readlines()
        self.samples = []

        for sample in tqdm(json_text):
            sample = sample.rstrip()
            sample = json.loads(sample)
            _inp_toked = RETOK.findall(sample['text'])
            _inp_toked_id = dictionary.t2v(_inp_toked)

            sample['text_vec'] = torch.tensor(_inp_toked_id, dtype=torch.long)

            # train and valid have different key names for target
            if dt == 'train':
                _tar_toked = RETOK.findall(sample['labels'][0]) + ['__end__']
            elif dt == 'valid':
                _tar_toked = RETOK.findall(sample['eval_labels'][0]) + ['__end__']

            _tar_toked_id = dictionary.t2v(_tar_toked)

            sample['target_vec'] = torch.tensor(_tar_toked_id, dtype=torch.long)

            self.samples.append(sample)

    def __getitem__(self, i):
        return self.samples[i]['text_vec'], self.samples[i]['target_vec']

    def __len__(self):
        return len(self.samples)

def pad_tensor(tensors, sort=True, pad_token=0):
    '''
    function to extend/pad with <pad_token> any tensors shorter than max-length
    tensor, as sequence tensors are going to be of varying length
    '''
    rows = len(tensors)
    lengths = [len(i) for i in tensors]
    max_t = max(lengths)

    output = tensors[0].new(rows, max_t)
    output.fill_(pad_token)  # 0 is a pad token here

    for i, (tensor, length) in enumerate(zip(tensors, lengths)):
        output[i,:length] = tensor
    return output, lengths

def argsort(keys, *lists, descending=False):
    '''
    Reorder each list in *lists by the sorted order of keys,
    either descending or ascending.
    :param iter keys: Keys to order by.
    :param list[list] lists: Lists to reorder by keys's new order.
                             Correctly handles lists and 1-D tensors.
    :param bool descending: Use descending order if true.
    :returns: The reordered items.
    '''
    ind_sorted = sorted(range(len(keys)), key=lambda k: keys[k])
    if descending:
        ind_sorted = list(reversed(ind_sorted))
    output = []
    for lst in lists:
        if isinstance(lst, torch.Tensor):
            output.append(lst[ind_sorted])
        else:
            output.append([lst[i] for i in ind_sorted])
    return output

def batchify(batch):
    '''
    batch function to be applied to tensors in training loop
    '''
    inputs = [i[0] for i in batch]
    labels = [i[1] for i in batch]

    input_vecs, input_lens = pad_tensor(inputs)
    label_vecs, label_lens = pad_tensor(labels)

    # sort only wrt inputs here for encoder packinng
    input_vecs, input_lens, label_vecs, label_lens = argsort(input_lens, input_vecs, input_lens,
                                                             label_vecs, label_lens, descending=True)

    return {
        'text_vecs': input_vecs,
        'text_lens': input_lens,
        'target_vecs': label_vecs,
        'target_lens': label_lens,
        'use_packed': True
    }

def calculate_perplexity(ce_loss):
    '''
    perplexity is essentially the exponentiation of entropy. it is a metric
    inversely proportional to the probability that the model assigns to a set of sequences; 
    i.e. a measurement of how well a probability model predicts a sample. intuitively, 
    perplexity measures the average rank of the true next-token, when tokens are ordered 
    by the model's conditional probabilities
    '''
    return 2**(ce_loss/np.log(2))

def format_perplexity(ppl):
    whole, dec = str(round(ppl, 10)).split('.')
    return '_perplexity_' + whole + '-' + dec


'''
***
REMEMBER: REACH OUT TO PROF. CHO OR TAs TO CONSULT ON BEST WAY TO CITE THEIR WORK
***
'''


class EncoderRNN(nn.Module):
    """Encodes the input context."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, pad_idx=0,
                 dropout=0, shared_lt=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.pad_idx = pad_idx

        if shared_lt is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size, pad_idx)
        else:
            self.embedding = shared_lt

        self.gru = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,)

    def forward(self, text_vec, text_lens, hidden=None, use_packed=True):
        embedded = self.embedding(text_vec)
        attention_mask = text_vec.ne(self.pad_idx)

        embedded = self.dropout(embedded)
        if use_packed is True:
            embedded = pack_padded_sequence(embedded, text_lens, batch_first=True)
        output, hidden = self.gru(embedded, hidden)
        if use_packed is True:
            output, output_lens = pad_packed_sequence(output, batch_first=True)

        return output, hidden, attention_mask

    
class DecoderRNN(nn.Module):
    """Generates a sequence of tokens in response to context."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, 0)
        
        self.gru = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,)
        
        self.attention = AttentionLayer(self.hidden_size, self.embed_size)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.longest_label = 100

    def forward(self, text_vec, decoder_hidden, encoder_states):
        emb = self.embedding(text_vec)
        emb = self.dropout(emb)
        seqlen = text_vec.size(1)
        encoder_output, encoder_hidden, attention_mask = encoder_states
        
        decoder_hidden = decoder_hidden
        output = []
        attn_w_log = []

        for i in range(seqlen):
            decoder_output, decoder_hidden = self.gru(emb[:,i,:].unsqueeze(1), decoder_hidden)
            
            # compute attention at each time step
            decoder_output_attended, attn_weights = self.attention(decoder_output, decoder_hidden, 
                                                                   encoder_output, attention_mask)
            output.append(decoder_output_attended)
            attn_w_log.append(attn_weights)
            
        output = torch.cat(output, dim=1).to(text_vec.device)
        scores = self.out(output)
        
        return scores, decoder_hidden, attn_w_log
    
    def decode_forced(self, ys, encoder_states, xs_lens):
        encoder_output, encoder_hidden, attention_mask = encoder_states

        batch_size = ys.size(0)
        target_length = ys.size(1)
        longest_label = max(target_length, self.longest_label)
        # expand to batch size
        starts = torch.Tensor([1]).long().to(self.embedding.weight.device)\
                 .expand(batch_size, 1).long()

        # Teacher forcing: Feed the target as the next input
        y_in = ys.narrow(1, 0, ys.size(1) - 1)
        decoder_input = torch.cat([starts, y_in], 1)
        decoder_output, decoder_hidden, attn_w_log = self.forward(decoder_input, encoder_hidden, 
                                                                  encoder_states)
        _, preds = decoder_output.max(dim=2)

        return decoder_output, preds, attn_w_log
    
    
class AttentionLayer(nn.Module):

    def __init__(self, hidden_size, embedding_size):
        super().__init__()
        input_dim = hidden_size

        self.linear_out = nn.Linear(hidden_size+input_dim, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, decoder_output, decoder_hidden, encoder_output, attention_mask):

        batch_size, seq_length, hidden_size = encoder_output.size()

        encoder_output_t = encoder_output.transpose(1, 2)

        attention_scores = torch.bmm(decoder_output, encoder_output_t).squeeze(1)

        attention_scores.masked_fill_((~attention_mask), -10e5)
        attention_weights = self.softmax(attention_scores)

        mix = torch.bmm(attention_weights.unsqueeze(1), encoder_output)

        combined = torch.cat((decoder_output.squeeze(1), mix.squeeze(1)), dim=1)

        output = self.linear_out(combined).unsqueeze(1)
        output = self.tanh(output)

        return output, attention_weights


def set_model_config(chat_dictionary, hidden_size=512, embedding_size=256, 
                     num_layers_enc=2, num_layers_dec=2, dropout=0.3, 
                     encoder_shared_lt=True):
    opts = {}
    opts['vocab_size'] = len(chat_dictionary)
    opts['hidden_size'] = hidden_size
    opts['embedding_size'] = embedding_size
    opts['num_layers_enc'] = num_layers_enc
    opts['num_layers_dec'] = num_layers_dec
    opts['dropout'] = dropout
    # I don't think is actually used but if it is uncomment
    # opts['encoder_shared_lt'] = encoder_shared_lt

    return opts

class seq2seq(nn.Module):
    """
    Generic seq2seq model with attention mechanism.
    """
    def __init__(self, opts):

        super().__init__()
        self.opts = opts
        
        self.decoder = DecoderRNN(vocab_size=self.opts['vocab_size'],
                                  embed_size=self.opts['embedding_size'],
                                  hidden_size=self.opts['hidden_size'],
                                  num_layers=self.opts['num_layers_dec'],
                                  dropout=self.opts['dropout'],)
        
        self.encoder = EncoderRNN(vocab_size=self.opts['vocab_size'],
                                  embed_size=self.opts['embedding_size'],
                                  hidden_size=self.opts['hidden_size'],
                                  num_layers=self.opts['num_layers_enc'],
                                  dropout=self.opts['dropout'],
                                  shared_lt=self.decoder.embedding)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

def build_dataloader(dataset, collate_func, batch_size, shuffle=True):
    loader = DataLoader(dataset, shuffle=shuffle, collate_fn=collate_func, batch_size=batch_size)
    return loader

'''
***

ANTICIPATE HAVING TO REWRITE BELOW TO FEED ARGS PARAM TO INITIATE CLASS
I.E. seq2seqTrainer(training_Args) 
MAYBE???
***
'''
class seq2seqTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, loss, optimizer,
                 num_epochs, device, models_dir):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss = loss
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        self.models_dir = models_dir

    def train_model(self):
        self.model.to(self.device)
        
        best_val_loss = np.inf
        sum_loss = 0
        sum_tokens = 0
        for epoch in range(self.num_epochs):
            print(EPOCH_LOG.format(dt.datetime.now(), (epoch + 1), self.num_epochs))
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                text_vecs = batch['text_vecs'].to(self.device)
                target_vecs = batch['target_vecs'].to(self.device)

                encoded = self.model.encoder(text_vecs, batch['text_lens'], use_packed=batch['use_packed'])

                decoder_output, preds, attn_w_log = self.model.decoder.decode_forced(target_vecs, encoded, batch['text_lens'])

                scores = decoder_output.view(-1, decoder_output.size(-1))

                loss = self.loss(scores, target_vecs.view(-1))
                sum_loss += loss.item()

                num_tokens = target_vecs.ne(0).long().sum().item()
                loss /= num_tokens

                sum_tokens += num_tokens

                loss.backward()
                self.optimizer.step()
                if step % 100 == 0:
                    print(STEP_LOG.format(dt.datetime.now(tz=TIMEZONE), calculate_perplexity(sum_loss/sum_tokens), step, len(self.train_dataloader)))
            avg_train_loss = sum_loss / sum_tokens
            val_loss = 0
            val_tokens = 0
            for step, batch in enumerate(self.valid_dataloader):
                self.model.eval()
                text_vecs = batch['text_vecs'].to(self.device)
                target_vecs = batch['target_vecs'].to(self.device)

                encoded = self.model.encoder(text_vecs, batch['text_lens'], use_packed=batch['use_packed'])

                decoder_output, preds, attn_w_log = self.model.decoder.decode_forced(target_vecs, encoded, batch['text_lens'])

                scores = decoder_output.view(-1, decoder_output.size(-1))

                loss = self.loss(scores, target_vecs.view(-1))

                num_tokens = target_vecs.ne(0).long().sum().item()

                val_tokens += num_tokens
                val_loss += loss.item()

            avg_val_loss = val_loss / val_tokens
            val_ppl = calculate_perplexity(avg_val_loss)
            self.scheduler.step(avg_val_loss)
            print('{} | validation perplexity achieved: {}'.format(dt.datetime.now(), val_ppl))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                formatted_ppl = format_perplexity(val_ppl)
                save_path = self.models_dir +'seq2seq_chatbot_epoch-'+str(epoch+1)+formatted_ppl+'.pt'
                torch.save(self.model.state_dict(), save_path)


##### BERT MODEL(S) #####

class BERTmodel(object):
    pass

class BERTtrainer(object):
    pass

"""
FIGURE OUT HOW BLEU SCORE WORKS
"""
