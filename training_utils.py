import re
import json
import torch
import copy
import numpy as np
import datetime as dt
import pickle as pkl
import pytz
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# from processing_utils import RETOK
# commented above, added below to run on GCP without needing to import
# processing_utils, i.e. installing lyrics_genius library
RETOK = re.compile(r'\w+|[^\w\s\[\]]|\n', re.UNICODE)
EPOCH_LOG = '{} | Epoch {} / {}'
STEP_LOG = '{} | Perplexity achieved: {} | Step {} / {}'
TIMEZONE = pytz.timezone('America/New_York')
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 
                         'pad_token': '<pad>', 'additional_special_tokens': ['<speaker1>', '<speaker2>']}
                         
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
                _tar_toked = RETOK.findall(sample['labels']) + ['__end__']
            elif dt == 'valid':
                _tar_toked = RETOK.findall(sample['eval_labels']) + ['__end__']

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

# encoder, decoder, and attention layer used here 
# were implemented by DS-GA1011 instructors 

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
        
        # decoder_hidden = decoder_hidden
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
                avg_train_loss = sum_loss / sum_tokens
                if step % 100 == 0:
                    print(STEP_LOG.format(dt.datetime.now(tz=TIMEZONE), calculate_perplexity(avg_train_loss), step, len(self.train_dataloader)))
            
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
            print('{} | Validation perplexity achieved: {}'.format(dt.datetime.now(), val_ppl))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                formatted_ppl = format_perplexity(val_ppl)
                save_path = self.models_dir +'seq2seq_chatbot_epoch-'+str(epoch+1)+formatted_ppl+'.pt'
                torch.save(self.model.state_dict(), save_path)


#### GPT2  ####
def add_special_tokens(model, tokenizer):
    original_num_tokens = len(tokenizer.encoder)
    tokens_to_add = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=tokens_to_add + original_num_tokens)

def tokenize_for_GPT2(dataset_file_path, tokenizer, stage='train', max_sentence_length=128):
    json_text = open(dataset_file_path, 'r').readlines()
    bos, eos, speaker1, speaker2, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    inputs = []
    attn_mask = []
    label_ids =[]

    for sample in tqdm(json_text):
        sample = json.loads(sample)
        text_tokenized = tokenizer.tokenize(sample['text'])
        if stage == 'valid':
            label_tokenized = tokenizer.tokenize(sample['eval_labels'])
        else:
            label_tokenized = tokenizer.tokenize(sample['labels'])

        if not len(text_tokenized) or not len(label_tokenized):
            continue
        else:
            if len(text_tokenized) > max_sentence_length:
                text_tokenized = text_tokenized[:max_sentence_length-5]
            if len(label_tokenized) > max_sentence_length:
                label_tokenized = label_tokenized[:max_sentence_length-5]

            encoded_text = tokenizer.encode(text_tokenized)
            encoded_text = [tokenizer.bos_token_id,]  + encoded_text + [tokenizer.eos_token_id,]
            encoded_label = tokenizer.encode(label_tokenized)
            encoded_label = [tokenizer.bos_token_id,] + encoded_label + [tokenizer.eos_token_id,]
            text_encoded_length = len(encoded_text)
            label_encoded_length = len(encoded_label)
            encoded_text += [pad_token_id,] * (max_sentence_length - text_encoded_length)
            encoded_label += [pad_token_id,] * (max_sentence_length - label_encoded_length)
            if pad_token_id in encoded_text:
                attn =  [1,] * (encoded_text.index(pad_token_id) + 1)
                attn += [0,] * (max_sentence_length - text_encoded_length - 1)
            else:
                attn = [1,] * max_sentence_length
            encoded_text = encoded_text[:max_sentence_length]
            encoded_label = encoded_text[:max_sentence_length]
            attn = attn[:max_sentence_length]

            inputs.append(encoded_text)
            attn_mask.append(attn)
            label_ids.append(encoded_label)


    inputs = torch.tensor(inputs, dtype=torch.long)
    attn_mask = torch.tensor(attn_mask, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)

    return TensorDataset(inputs, attn_mask, label_ids)

class GPT2model(object):
    def __init__(self, num_epochs, batch_size, device, models_dir):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.models_dir = models_dir

    def train(self, train_dataset, valid_dataset, optimizer):
        self.model.to(self.device)
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = RandomSampler(valid_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=self.batch_size)

        best_val_loss = np.inf
        loss_set = []
        for epoch in range(self.num_epochs):
            print(EPOCH_LOG.format(dt.datetime.now(), (epoch + 1), self.num_epochs))
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_masks, label_ids = batch
                loss = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=label_ids)[0]

                loss_set.append(loss.item())
                loss.backward()
                optimizer.step()

                avg_train_loss = np.mean(loss_set)
                if step % 100 == 0:
                    print(STEP_LOG.format(dt.datetime.now(tz=TIMEZONE), calculate_perplexity(avg_train_loss), step, len(train_dataloader)))

            val_loss_set = []
            for step, batch in enumerate(valid_dataloader):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_masks, label_ids = batch
                loss = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=label_ids)[0]
                val_loss_set.append(loss.item())

            avg_val_loss = np.mean(val_loss_set)
            val_ppl = calculate_perplexity(avg_val_loss)
            print('{} | Validation perplexity achieved: {}'.format(dt.datetime.now(), val_ppl))
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                formatted_ppl = format_perplexity(val_ppl)
                save_path = self.models_dir +'GPT2_chatbot_epoch-'+str(epoch+1)+formatted_ppl+'.pt'
                self.model.save_pretrained(save_path)


class pretrained_model(object):
    def __init__(self, model_name, num_epochs, batch_size, device, models_dir, patience=5):
        self.model_name = model_name
        if self.model_name == 'BERT':
            self.encoder = BertGenerationEncoder.from_pretrained('bert-base-uncased', bos_token_id=101, eos_token_id=102) 
            self.decoder = BertGenerationDecoder.from_pretrained('bert-base-uncased', add_cross_attention=True,
                                                                 is_decoder=True, bos_token_id=101, eos_token_id=102)
            self.model = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        elif self.model_name == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        else:
            raise ValueError('Invalid model seleciton.')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.models_dir = models_dir
        self.patience = patience

    def tokenize_data(self, dataset_file_path, stage='train', max_sentence_length=128):
        max_sentence_length = max_sentence_length - 2
        json_text = open(dataset_file_path, 'r').readlines()
        if self.model_name == 'BERT':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            input_ids_encode = []
            attention_masks_encode = []
            input_ids_decode = []
            attention_masks_decode = []
            lm_labels = []

            for sample in tqdm(json_text):
                sample = json.loads(sample)
                text_tokenized = tokenizer.tokenize(sample['text'])
                if stage == 'valid':
                    labels_tokenized = tokenizer.tokenize(sample['eval_labels'])
                else:
                    labels_tokenized = tokenizer.tokenize(sample['labels'])

                text_tokenized_length = len(text_tokenized)
                labels_tokenized_length = len(labels_tokenized)

                if text_tokenized_length > max_sentence_length:
                    text_tokenized = text_tokenized[:max_sentence_length]
                    text_tokenized_length = max_sentence_length

                if labels_tokenized_length > max_sentence_length:
                    labels_tokenized = labels_tokenized[:max_sentence_length]
                    labels_tokenized_length = max_sentence_length
                try:
                    input_id_enc = tokenizer.encode(text_tokenized)
                    input_id_enc += [0,] * (max_sentence_length - text_tokenized_length)
                    input_id_dec = tokenizer.encode(labels_tokenized)
                    input_id_dec += [0,] * (max_sentence_length - labels_tokenized_length)
                except:
                    print(text_tokenized)
                    print(labels_tokenized)

                assert len(input_id_enc)==(max_sentence_length + 2)
                assert len(input_id_dec)==(max_sentence_length + 2)
                lm_label = [-100 if token==0 else token for token in input_id_dec]

                attn_mask_enc = [float(i>0) for i in input_id_enc]
                attn_mask_dec = [float(i>0) for i in input_id_dec]

                input_ids_encode.append(input_id_enc)
                input_ids_decode.append(input_id_dec)
                attention_masks_encode.append(attn_mask_enc)
                attention_masks_decode.append(attn_mask_dec)
                lm_labels.append(lm_label)

            input_ids_encode = torch.tensor(input_ids_encode, dtype=torch.long)
            attention_masks_encode = torch.tensor(attention_masks_encode, dtype=torch.long)
            input_ids_decode = torch.tensor(input_ids_decode, dtype=torch.long)
            attention_masks_decode = torch.tensor(attention_masks_decode, dtype=torch.long)
            lm_labels = torch.tensor(lm_labels, dtype=torch.long)

            return TensorDataset(input_ids_encode, attention_masks_encode, input_ids_decode, attention_masks_decode, lm_labels)

        elif self.model_name == 'GPT2':
            # TO-DO
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            print('TO-DO')

    def train_step(self, model, train_loader, optimizer):
        loss_set = []
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(self.device) for t in batch)
            if self.model_name == 'BERT':
                input_ids_encode, attention_masks_encode, input_ids_decode, attention_masks_decode, lm_labels = batch
                loss = model(input_ids=input_ids_encode, decoder_input_ids=input_ids_decode,
                             attention_mask=attention_masks_encode, decoder_attention_mask=attention_masks_decode,
                             labels=lm_labels)[0]
            elif self.model_name == 'GPT2':
                print('TO-DO :)')

            loss_set.append(loss.item())
            loss.backward()
            optimizer.step()
            avg_train_loss = np.mean(loss_set)
            if step % 100 == 0:
                print(STEP_LOG.format(dt.datetime.now(tz=TIMEZONE), calculate_perplexity(avg_train_loss),
                                      step, len(train_loader)))

    def validation_step(self, model, val_loader, best_val_loss, epoch):
        model.eval()
        val_loss_set = []
        for step, batch in enumerate(val_loader):
            batch = tuple(t.to(self.device) for t in batch)
            if self.model_name == 'BERT':
                input_ids_encode, attention_masks_encode, input_ids_decode, attention_masks_decode, lm_labels = batch
                loss = model(input_ids=input_ids_encode, decoder_input_ids=input_ids_decode,
                             attention_mask=attention_masks_encode, decoder_attention_mask=attention_masks_decode,
                             labels=lm_labels)[0]
            elif self.model_name == 'GPT2':
                print('TO-DO')

            val_loss_set.append(loss.item())
        avg_val_loss = np.mean(val_loss_set)
        val_ppl = calculate_perplexity(avg_val_loss)
        print('{} | Validation perplexity achieved: {:.4}'.format(dt.datetime.now(), val_ppl))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            formatted_ppl = format_perplexity(val_ppl)
            save_path = self.models_dir +'bert2bert_chatbot_epoch-'+str(epoch+1)+formatted_ppl+'.pt'
            self.model.save_pretrained(save_path)
        return best_val_loss

    def train(self, train_dataset, valid_dataset, optimizer, step_size, gamma):
        self.model.to(self.device)
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = RandomSampler(valid_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=self.batch_size)

        best_val_loss = np.inf
        loss_set = []
        patience_counter = 0
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        for epoch in range(self.num_epochs):
            print(EPOCH_LOG.format(dt.datetime.now(), (epoch + 1), self.num_epochs))
            self.model.train()
            self.train_step(self.model, train_dataloader, optimizer)
            previous_val_loss = best_val_loss
            best_val_loss = self.validation_step(self.model, valid_dataloader, best_val_loss, epoch)
            if patience_counter > self.patience:
                print('Best val loss: {:.4}'.format(best_val_loss))
                break
            if best_val_loss < previous_val_loss:
                patience_counter = 0 
            else:
                patience_counter += 1

