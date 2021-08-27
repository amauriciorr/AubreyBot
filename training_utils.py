import re
import json
import pytz
import torch
import numpy as np
from datetime import datetime
from copy import copy
import pickle as pkl
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModelForPreTraining, AutoTokenizer, AdamW

# from processing_utils import RETOK
# commented above, added below to run on GCP without needing to import
# processing_utils, i.e. installing lyrics_genius library
RETOK = re.compile(r'\w+|[^\w\s\[\]]|\n', re.UNICODE)
EPOCH_LOG = '{} | Epoch {} / {}'
STEP_LOG = '{} | Perplexity {:.4} | Step {} / {}'
VERBOSE_STEP_LOG = '{} | Avg. Loss : {:.4} | Perplexity {:.4} | Step {} / {}'
TIMEZONE = pytz.timezone('America/New_York')

def calculate_perplexity(ce_loss):
    '''
    perplexity is essentially the exponentiation of entropy. it is a metric
    inversely proportional to the probability that the model assigns to a set of sequences; 
    i.e. a measurement of how well a probability model predicts a sample. intuitively, 
    perplexity measures the average rank of the true next-token, when tokens are ordered 
    by the model's conditional probabilities
    '''  
    perplexity = float(2**(ce_loss/np.log(2)))
    return perplexity

def format_perplexity(ppl):
    try:
        whole, dec = str(round(ppl, 10)).split('.')
        formatted_output = '_perplexity_' + whole + '-' + dec
    except:
        formatted_output = '_perplexity_inf'
    return formatted_output

class pretrained_model(object):
    def __init__(self, model_name, num_epochs, batch_size, sentence_length, device, models_dir, patience=5):
        self.model_name = model_name
        # self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.model = AutoModelForPreTraining.from_pretrained(model_name)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.models_dir = models_dir
        self.patience = patience
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_data(self, dataset_file_path, max_sentence_length, stage='train'):
        json_text = open(dataset_file_path, 'r').readlines()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_ids_encode = []
        labels = []
        attention_masks = []
        for sample in tqdm(json_text):
            sample = json.loads(sample)
            if stage == 'valid':
                label = sample['text'] + ' ' + sample['eval_labels']
                encoded_label = self.tokenizer.encode(label, padding = 'max_length', 
                                                      truncation = True, max_length = max_sentence_length)
            else:
                label = sample['text'] + ' ' + sample['labels']
                encoded_label = self.tokenizer.encode(label, padding = 'max_length', 
                                                      truncation = True, max_length = max_sentence_length)
            encoded = self.tokenizer.encode_plus(sample['text'], padding = 'max_length',
                                                 max_length = max_sentence_length, truncation = True)
            input_ids_encode.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(encoded_label)
        input_ids_encode = torch.tensor(input_ids_encode, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return TensorDataset(input_ids_encode, attention_masks, labels)

    def train_step(self, model, train_loader, optimizer):
        loss_set = []
        
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_masks, labels = batch
            # loss = model(input_ids, labels = input_ids, attention_mask = attention_masks)[0]
            loss = model(input_ids, labels = labels, attention_mask = attention_masks)[0]
            optimizer.zero_grad()
            loss_set.append(loss.item())
            loss.backward()
            optimizer.step()
            avg_train_loss = np.mean(loss_set)
            if step % 100 == 0:
                print(VERBOSE_STEP_LOG.format(datetime.now(tz=TIMEZONE), avg_train_loss, 
                      calculate_perplexity(avg_train_loss), step, len(train_loader)))

    def validation_step(self, model, val_loader, best_val_loss, epoch):
        model.eval()
        val_loss_set = []
        for step, batch in enumerate(val_loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_masks, labels = batch
            # loss = model(input_ids, labels = input_ids, attention_mask = attention_masks)[0]
            loss = model(input_ids, labels = labels, attention_mask = attention_masks)[0]
            val_loss_set.append(loss.item())
        avg_val_loss = np.mean(val_loss_set)
        val_ppl = calculate_perplexity(avg_val_loss)
        print('{} | Validation perplexity achieved: {:.4}'.format(datetime.now(tz=TIMEZONE), val_ppl))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            formatted_ppl = format_perplexity(val_ppl)
            save_path = self.models_dir +self.model_name+'_chatbot_epoch-'+str(epoch+1)+formatted_ppl+'.pt'
            self.model.save_pretrained(save_path)
        return best_val_loss

    def train(self, train_dataset, valid_dataset, optimizer, step_size, gamma):
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.model.to(self.device)
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = RandomSampler(valid_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=self.batch_size)

        best_val_loss = np.inf
        loss_set = []
        patience_counter = 0

        for epoch in range(self.num_epochs):
            print(EPOCH_LOG.format(datetime.now(tz=TIMEZONE), (epoch + 1), self.num_epochs))
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
        scheduler.step()
