import torch
from chat_utils import *
from training_utils import * 
from args import get_chat_args

# TO-DO
# args should specify which approach used to generate responses
# i.e. beam search or nucleus sampling



if __name__ == '__main__':
    chatbot_args = get_chat_args()
    current_device = torch.device("cuda" if chatbot_args.with_cuda else "cpu")
    chat_dict = ChatDictionary(chatbot_args.vocab_path)
    opts = set_model_config(chat_dict)
    model_checkpoint = torch.load(chatbot_args.model_path, map_location = current_device)
    chatbot = seq2seq(opts)
    chatbot.load_state_dict(model_checkpoint)
    chatbot.to(current_device)
    chatbot.eval()
    start_rapbot(chatbot, chat_dict, chatbot_args.top_p, current_device, transformer = False)