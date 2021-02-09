import torch
from chat_utils import *
from training_utils import * 
from args import get_chat_args


if __name__ == '__main__':
    chatbot_args = get_chat_args()
    current_device = torch.device("cuda" if chatbot_args.with_cuda else "cpu")
    chatbot = AutoModelForPreTraining.from_pretrained(chatbot_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(chatbot_args.pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    transfer_learning_bot(chatbot, tokenizer, chatbot_args.max_sentence_length,
                          chatbot_args.top_k, chatbot_args.top_p)
