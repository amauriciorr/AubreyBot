import os
import pickle as pkl
from args import get_train_args
from training_utils import *
import logging

INIT_MODEL = '{} | Initializing model'
TRAIN_MSG = '{} | Training model'

def train():
    args = get_train_args()
    current_device = torch.device("cuda" if args.with_cuda else "cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.use_logging:
        log_filename = str(dt.datetime.now(tz=TIMEZONE))
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler(log_filename+'.log', 'a'))
        print = logger.info

    if not args.pretrained_model:
        chat_dict = ChatDictionary('./word_counts_dict.p')
        train_dataset = ChatDataset(chat_dict, './train_lyrics.jsonl')
        valid_dataset = ChatDataset(chat_dict, './valid_lyrics.jsonl', 'valid')
        train_dataloader = build_dataloader(train_dataset, batchify, args.batch_size)
        valid_dataloader = build_dataloader(valid_dataset, batchify, args.batch_size,
                                            shuffle=False)
        opts = set_model_config(chat_dict, args.hidden_size, args.embedding_size,
                                args.num_layers_enc, args.num_layers_dec,
                                args.dropout)
        model = seq2seq(opts)
        model.to(current_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                     weight_decay=args.weight_decay, amsgrad=True)
        model_trainer = seq2seqTrainer(model, train_dataloader, valid_dataloader, criterion,
                                       optimizer, args.num_epochs, current_device, 
                                       args.save_dir)
        model_trainer.train_model()

    else:
        model = pretrained_model(args.pretrained_model, args.num_epochs, args.batch_size, current_device,
                                 args.save_dir, args.patience)
        train_dataset = model.tokenize_data('./train_lyrics.jsonl', args.max_sentence_length)
        valid_dataset = model.tokenize_data('./valid_lyrics.jsonl', 'valid', args.max_sentence_length)
        optimizer = AdamW(model.model.parameters(), lr=args.learning_rate, eps=args.eps)
        model.train(train_dataset, valid_dataset, optimizer)
        

if __name__ == "__main__":
    train()
    # training_args = get_train_args()
    # current_device = torch.device("cuda" if training_args.with_cuda else "cpu")
    # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')



    # if not training_args.use_BERT and not training_args.use_GPT2:
    #     chat_dict = ChatDictionary('./word_counts_dict.p')

    #     print('{} | Creating dataset...\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     train_dataset = ChatDataset(chat_dict, './train_lyrics.jsonl')
    #     valid_dataset = ChatDataset(chat_dict, './valid_lyrics.jsonl', 'valid')

    #     print('\n {} | Creating dataloader...\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     train_dataloader = build_dataloader(train_dataset, batchify, training_args.batch_size)
    #     valid_dataloader = build_dataloader(valid_dataset, batchify, training_args.batch_size,
    #                                         shuffle=False)

    #     opts = set_model_config(chat_dict, training_args.hidden_size, training_args.embedding_size,
    #                             training_args.num_layers_enc, training_args.num_layers_dec,
    #                             training_args.dropout)

    #     print('\n {} | Initializing seq2seq model...\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     model = seq2seq(opts)
    #     model.to(current_device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate, 
    #                                  weight_decay=training_args.weight_decay, amsgrad=True)
    #     model_trainer = seq2seqTrainer(model, train_dataloader, valid_dataloader, criterion,
    #                                    optimizer, training_args.num_epochs, current_device, 
    #                                    training_args.save_dir)

    #     print('\n {} | Begin training model...\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     model_trainer.train_model()
    # elif training_args.use_BERT:
    #     print('\n {} | Tokenizing data\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     train_dataset = tokenize_for_BERT('./train_lyrics.jsonl', 
    #                                       max_sentence_length=training_args.max_sentence_length)
    #     valid_dataset = tokenize_for_BERT('./valid_lyrics.jsonl', 'valid', 
    #                                       max_sentence_length=training_args.max_sentence_length)

    #     print('\n{} | Initializing BERT...\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     bert2bert_model = BERT2BERT(training_args.num_epochs, training_args.batch_size, current_device,
    #                                 training_args.save_dir)
    #     param_optimizer = list(bert2bert_model.model.named_parameters())
    #     no_decay = ['bias', 'gamma', 'beta']
    #     # seperate decay
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #         'weight_decay_rate': 0.01},
    #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #         'weight_decay_rate': 0.0}]

    #     optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, 
    #                       eps=training_args.eps)

    #     print('\n {} | Training BERT2BERT...\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     bert2bert_model.train_bert(train_dataset, valid_dataset, criterion, optimizer)

    # elif training_args.use_GPT2:
    #     print('\n {} | Tokenizing data\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     train_dataset = tokenize_for_GPT2('./train_lyrics.jsonl', 
    #                                       max_sentence_length=training_args.max_sentence_length)
    #     valid_dataset = tokenize_for_GPT2('./valid_lyrics.jsonl', 'valid',
    #                                       max_sentence_length=training_args.max_sentence_length)
    #     print('\n{} | Initializing GPT2...\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     gpt2_model = GPT2model(training_args.num_epochs,  training_args.batch_size, current_device,
    #                            training_args.save_dir)
    #     optimizer = AdamW(gpt2_model.model.parameters(), lr=training_args.learning_rate)
    #     print('\n {} | Training GPT2...\n'.format(dt.datetime.now(tz=TIMEZONE)))
    #     gpt2_model.train(train_dataset, valid_dataset, optimizer)
