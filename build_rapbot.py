from args import * 
from training_utils import *
from processing_utils import * 

if __name__ == "__main__":
    preprocess_args = get_setup_args()
    get_and_process_songs(preprocess_args)
        '''
        TO ADD
        1) FUNCTION FOR CREATING ChatDataset() + ChatDataset() CLASS OBJECTS AND SAVING THEM OR FEEDING THEM FOR USE IN TRAINING CLASS/WORKFLOW
        2) ADD ABOVE AS PRE-SAVED FILES FOR SKIPPING DOWNLOAD/PREPROCESS SET???
        3) TRAINING MODEL PORTION
        '''
