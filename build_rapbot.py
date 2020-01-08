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

        '''
        TO ADD
 		1) model architecture
 			encoder
 			decoder
 			attn mechanism
        2) TRAINING MODEL PORTION/ training loop class

        TO TEST:
        1) ADDING BERT
        2) LSTM VS GRU 
        '''
