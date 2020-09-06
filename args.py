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
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load artist lyrics JSON.')
    args = parser.parse_args()

    return args

def get_train_args():
    parser = argparse.ArgumentParser('Train a chatbot on lyrics')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to train model.')
    parser.add_argument('--with_cuda',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Use CUDA when available.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Controls how much we are adjusting the weights of our network with\
                              respect to the loss gradient.')
    # parser.add_argument('--verbose',
    #                    type=lambda s: s.lower().startswith('t'),
    #                    default=True,
    #                    help='Print logging details to console. Includes epoch number, 
    #                          loss, perplexity.')
    #LOOK INTO WHAT EXACTLY I WANT TO INCLUDE IN VERBOSE LOGGER ???
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Size of batch, i.e. size of data partitions')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=512,
                        help='The number of features in the hidden state h, i.e. the number of\
                         nodes or neurons in the hidden layer')
    parser.add_argument('--embedding_size',
                        type=int,
                        default=256,
                        help='Vector size for embedding representation, i.e. dimension of embedding\
                              vector.')
    parser.add_argument('--num_layers_enc',
                        type=int,
                        default=2,
                        help='Number of recurrent layers, e.g. a value of 2 would mean stacking two\
                              GRUs together')
    parser.add_argument('--num_layers_dec',
                        type=int,
                        default=2,
                        help='Number of recurrent layers, e.g. a value of 2 would mean stacking two\
                              GRUs together')
    parser.add_argument('--dropout',
                        type=int,
                        default=0.3,
                        help='Percentage of nodes to randomly ignore as part of Dropout method.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./models/',
                        help='Directory to save models.')
    args = parser.parse_args()
    return args

    '''
    ***
    TO-DO
    ADD CONFIG FOR VARIOUS PARAMETERS
    OPTIMIZER
    LOSS FN
    ETC.
    ***
    '''