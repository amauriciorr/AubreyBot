import os
import pickle as pkl
from args import get_train_args
from training_utils import *
import logging

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

    model = pretrained_model(args.pretrained_model, args.num_epochs, args.batch_size, args.max_sentence_length,
                             current_device, args.save_dir, args.patience)
    train_dataset = model.tokenize_data('./train_lyrics.jsonl', args.max_sentence_length)
    valid_dataset = model.tokenize_data('./valid_lyrics.jsonl', args.max_sentence_length, 'valid')
    optimizer = AdamW(model.model.parameters(), lr=args.learning_rate, eps=args.eps,
                      weight_decay=args.weight_decay)
    model.train(train_dataset, valid_dataset, optimizer, args.step_size, args.gamma)     

if __name__ == "__main__":
    train()
