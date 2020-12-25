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
    if not chatbot_args.pretrained_model:
        chat_dict = ChatDictionary(chatbot_args.vocab_path)
        opts = set_model_config(chat_dict)
        model_checkpoint = torch.load(chatbot_args.model_path, map_location = current_device)
        chatbot = seq2seq(opts)
        chatbot.load_state_dict(model_checkpoint)
        chatbot.to(current_device)
        chatbot.eval()
        start_rapbot(chatbot, chat_dict, chatbot_args.top_p, current_device, transformer = False)
    else:
        chatbot = AutoModelForPreTraining.from_pretrained(chatbot_args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(chatbot_args.pretrained_model)
        tokenizer.pad_token = tokenizer.eos_token
        transfer_learning_bot(chatbot, tokenizer, chatbot_args.max_sentence_length,
                              chatbot_args.top_k, chatbot_args.top_p, chatbot_args.repetition_penalty,
                              chatbot_args.no_repeat_ngram_size, chatbot_args.length_penalty)
