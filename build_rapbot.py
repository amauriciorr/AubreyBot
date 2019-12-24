from args import * 
from training_utils import *
from processing_utils import * 


if __name__ == "__main__":
    preprocess_args = get_setup_args()
    get_and_process_songs(preprocess_args)

