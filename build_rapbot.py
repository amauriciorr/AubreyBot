import os
import pickle as pkl
from args import get_train_args
from training_utils import *

if __name__ == "__main__":
    training_args = get_train_args()
    current_device = torch.device("cuda" if training_args.with_cuda else "cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if not os.path.exists(training_args.save_dir):
        os.makedirs(training_args.save_dir)

    if not training_args.use_BERT:
        chat_dict = ChatDictionary('./word_counts_dict.p')

        print('{} | Creating dataset...\n'.format(dt.datetime.now(tz=TIMEZONE)))
        train_dataset = ChatDataset(chat_dict, './train_lyrics.jsonl')
        valid_dataset = ChatDataset(chat_dict, './valid_lyrics.jsonl', 'valid')

        print('\n {} | Creating dataloader...\n'.format(dt.datetime.now(tz=TIMEZONE)))
        train_dataloader = build_dataloader(train_dataset, batchify, training_args.batch_size)
        valid_dataloader = build_dataloader(valid_dataset, batchify, training_args.batch_size,
                                            shuffle=False)

        opts = set_model_config(chat_dict, training_args.hidden_size, training_args.embedding_size,
                                training_args.num_layers_enc, training_args.num_layers_dec,
                                training_args.dropout)

        print('\n {} | Initializing seq2seq model...\n'.format(dt.datetime.now(tz=TIMEZONE)))
        model = seq2seq(opts)
        model.to(current_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate, 
                                     weight_decay=training_args.weight_decay, amsgrad=True)
        model_trainer = seq2seqTrainer(model, train_dataloader, valid_dataloader, criterion,
                                       optimizer, training_args.num_epochs, current_device, 
                                       training_args.save_dir)

        print('\n {} | Begin training model...\n'.format(dt.datetime.now(tz=TIMEZONE)))
        model_trainer.train_model()
    else:
        print('\n {} | Tokenizing data\n'.format(dt.datetime.now(tz=TIMEZONE)))
        train_dataset = tokenize_for_BERT('./train_lyrics.jsonl', 
                                          max_sentence_length=training_args.max_sentence_length)
        valid_dataset = tokenize_for_BERT('./valid_lyrics.jsonl', 'valid', 
                                          max_sentence_length=training_args.max_sentence_length)

        print('\n{} | Initializing BERT...\n'.format(dt.datetime.now(tz=TIMEZONE)))
        bert2bert_model = BERT2BERT(training_args.num_epochs, training_args.batch_size, current_device,
                                    training_args.save_dir)
        optimizer = torch.optim.Adam([p for p in bert2bert_model.model.parameters() if p.requires_grad],
                                     lr=training_args.learning_rate, eps=training_args.eps,
                                     weight_decay=training_args.weight_decay)

        print('\n {} | Training BERT2BERT...\n'.format(dt.datetime.now(tz=TIMEZONE)))
        bert2bert_model.train_bert(train_dataset, valid_dataset, criterion, optimizer)


