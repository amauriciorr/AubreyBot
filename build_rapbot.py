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
