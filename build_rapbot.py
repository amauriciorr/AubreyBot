import pickle as pkl
from args import * 
from training_utils import *
from processing_utils import * 

if __name__ == "__main__":
    preprocess_args = get_setup_args()
    get_and_process_songs(preprocess_args)
    chat_dict = ChatDictionary('./word_counts_dict.p')
    train_dataset = ChatDataset(chat_dict,'./train_lyrics.jsonl')
    valid_dataset = ChatDataset(chat_dict,'./valid_lyrics.jsonl', 'valid')
    training_args = get_train_args()
    train_dataloader = build_dataloader(train_dataset, batchify, training_args.batch_size)
    valid_dataloader = build_dataloader(valid_dataset, batchify, training_args.batch_size)


        '''
        TO ADD
        1) function that wraps entire training process into one line, i.e. feed training args to function
 		2) model architecture (standard + BERT)
 			encoder
 			decoder
 			attn mechanism
 		3) default model config as part of args.py
        4) TRAINING MODEL PORTION/ training loop class

        TO TEST:
        1) ADDING BERT
        2) LSTM VS GRU 
        '''
