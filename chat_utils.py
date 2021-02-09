import re
import time
import textwrap
from copy import copy
import torch.nn.functional as F
from training_utils import *

BASH_FORMATTING = {
                   'PURPLE': '\033[95m',
                   'CYAN': '\033[96m',
                   'DARKCYAN': '\033[36m',
                   'BLUE': '\033[94m',
                   'GREEN': '\033[92m',
                   'YELLOW': '\033[93m',
                   'RED':'\033[91m',
                   'BOLD': '\033[1m',
                   'UNDERLINE': '\033[4m',
                   'END': '\033[0m'
}

def transfer_learning_bot(model, tokenizer, max_length, top_k, top_p):
    '''
    for chatbot trained using transfer learning
    '''
    input_sentence = input('User >> ')
    input_sentence = input_sentence.lower()
    context = copy(input_sentence)
    input_sentence = tokenizer.encode(input_sentence, truncation = True, max_length = 128, return_tensors = 'pt')
    continue_convo = True
    while continue_convo:
        print(bash_format_text('Typing...', 'YELLOW', 'BOLD'), end='\r' )
        uni_temp = round(torch.rand(1).clamp(0.1).item(), 2)
        repeat_penalty = round((torch.rand(1) * 5).clamp(1).item(), 2)
        ngram =  int(np.random.choice([2,3,4], 1)[0])
        bot_reply = model.generate(input_sentence, max_length = max_length, top_k = top_k, top_p = top_p, temperature = uni_temp, 
                                   repetition_penalty = repeat_penalty, skip_special_tokens = True,
                                   no_repeat_ngram_size=ngram, pad_token_id = tokenizer.eos_token_id)
                                   # length_penalty=length_penalty)
        bot_reply = tokenizer.decode(bot_reply.squeeze()).replace('<|endoftext|>', '')
        bot_reply = textwrap.fill(bot_reply, width=75)
        print(bash_format_text('Aubrey: {}'.format(bot_reply), 'YELLOW', 'BOLD'))
        response = input('User >> ')
        if (response == 'q' or response == 'quit' or response == 'exit'):
            continue_convo = False
        input_sentence = tokenizer.encode(response.lower(), truncation= True, max_length = 128, return_tensors = 'pt')
