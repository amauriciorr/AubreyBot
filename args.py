import argparse

def get_setup_args():
    parser = argparse.ArgumentParser('Download and pre-process rap lyrics')
    parser.add_argument("--artist_name",
                        type=str,
                        help="Name of rapper to get lyrics for.")
    parser.add_argument("--download_lyrics",
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help="If lyrics JSON file already created, avoid downloading and start with building vocab and tokenizing.")
    parser.add_argument("--load_path",
                        type=str,
                        default=None,
                        help="Path to load artist lyrics JSON.")
    args = parser.parse_args()

    return args

def get_train_args():
    parser = argparse.ArgumentParser('Train a chatbot on lyrics')
    parser.add_argument()
    '''
    TO-DO
    ADD CONFIG FOR VARIOUS PARAMETERS
    OPTIMIZER
    LOSS FN
    ETC.
    '''