import argparse

def get_setup_args():
    parser = argparse.ArgumentParser('Download and pre-process rap lyrics')
    parser.add_argument('--artist_name',
                        type=str,
                        help='Name of rapper to get lyrics for.')
    parser.add_argument('--download_lyrics',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='If lyrics JSON file already created, avoid downloading and start with\
                              building vocab and tokenizing.')
    parser.add_argument('--download_only',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Forgo preprocessing dataset, only download Genius lyrics.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load artist lyrics JSON.')
    parser.add_argument('--lookback',
                        type=int,
                        default=5,
                        help='Number of lines to include as context.')
    args = parser.parse_args()

    return args

def get_train_args():
    parser = argparse.ArgumentParser('Train a chatbot on lyrics')
    parser.add_argument('--with_cuda',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Use CUDA when available.')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default=None,
                        help='LM model head to use for transfer learning, e.g. gpt2-medium')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Number of epochs to train model.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help='Controls how much we are adjusting the weights of our network with\
                              respect to the loss gradient.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='L2 regularization penalty. Causes weights to exponentially decay\
                              to zero.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Size of batch, i.e. size of data partitions')
    parser.add_argument('--eps',
                        type=float,
                        default=1e-08,
                        help='Adam\'s epsilon for numerical stability.')
    parser.add_argument('--max_sentence_length',
                        type=int,
                        default=128,
                        help='Max sequence length for BERT tokenizing and subsequent encoding. Note: \
                              pretrained BERT can only handle up to 512 tokens per sequence at once.')
    parser.add_argument('--patience', 
                        type=int,
                        default=5,
                        help='Max patience for whether or not to continue training process.')
    parser.add_argument('--step_size',
                        type=int,
                        default=2,
                        help='Period of learning rate decay, i.e. decay the learning rate of each\
                              parameter group by gamma every step_size epochs')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.1,
                        help='Multiplicative factor for learning rate decay.')
    parser.add_argument('--use_logging',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Save stdout to a log file.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./models/',
                        help='Directory to save models.')
    args = parser.parse_args()
    return args

def get_chat_args():
    parser = argparse.ArgumentParser('Preferences for talking with your chatbot.')
    parser.add_argument('--with_cuda',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Use CUDA when available.')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default=None,
                        help='Load pretrained model architecture for chatbot. Currently \
                              \'BERT\' and \'GPT2\' are supported')
    parser.add_argument('--temperature',
                        type=float,
                        default=1.0,
                        help='Value used for next token probabilities.')
    parser.add_argument('--top_p',
                        type=float,
                        default=0.9,
                        help='Top-p filter for output predicted probabilities.')
    parser.add_argument('--top_k',
                        type=int,
                        default=50,
                        help='Number of highest probability vocabulary tokens to keep.')
    parser.add_argument('--model_path',
                        type=str,
                        help='Specify path to trained chatbot model.')
    parser.add_argument('--length_penalty',
                        type=float,
                        default=1.0,
                        help='Exponential penalty to the length. 1.0 means no penalty. Set to values\
                              < 1.0 in order to encourage the model to generate shorter sequences, to \
                              a value > 1.0 in order to encourage the model to produce longer sequences.')
    parser.add_argument('--max_sentence_length',
                        type=int,
                        default=64,
                        help='The maximum length of the sequence to be generated.')
   
    args = parser.parse_args()
    return args
