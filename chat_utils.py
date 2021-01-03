import re
import time
import textwrap
from copy import copy
import torch.nn.functional as F
from training_utils import *

# T0-D0
# add beam search class?
# include nucleus sampling functions
# for embedding representation
# already built-in for BERT
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

def top_p_filter(logits, p, filter_val=-float('inf')):
     logits_sorted, logits_sorted_idx = torch.sort(logits, descending = True)
     cumulative_probs = torch.cumsum(F.softmax(logits_sorted, dim=-1), dim=-1)
     to_remove = cumulative_probs > p
     if to_remove.all():
         to_remove[..., 0] = False
     else:
         to_remove[..., 1:] = to_remove[..., :-1].clone()
     idx_to_remove = logits_sorted_idx[to_remove]
     logits[..., idx_to_remove] = filter_val
     return logits

def nucleus_sample(model, input_tensor, chat_dictionary, top_p, device, make_batch = True, transformer = False):
    END_IDX = chat_dictionary.word2ind['__end__']
    if make_batch:
        batch = batchify([input_tensor])
    else:
        batch = input_tensor
    text_vecs = batch['text_vecs']
    text_vecs = text_vecs.to(device)
    text_lens = batch['text_lens']
    bsz = text_vecs.size(0)

    model.encoder.eval()
    model.decoder.eval()
    if transformer:
        encoded = model.encoder(text_vecs)
    else:
        encoded = model.encoder(text_vecs, text_lens, use_packed=batch['use_packed'])
    encoder_output, encoder_hidden, attention_mask = encoded
    
    decoder_hidden = encoder_hidden
    starts = torch.Tensor([1]).long().to(device).expand(bsz, 1).long()
    done = [False for _ in range(bsz)]
    total_done = 0
    predictions = [starts]
    decoder_input = starts
    log_prob = 0.0
    for i in range(model.decoder.longest_label):
        decoder_output, decoder_hidden, attn_wts = model.decoder(decoder_input, decoder_hidden, encoded)
        orig_probs =  F.softmax(decoder_output, dim = -1)
        decoder_out_filtered = top_p_filter(decoder_output, p = top_p)
        top_p_probs = F.softmax(decoder_out_filtered, dim=-1)
        predicted = torch.multinomial(top_p_probs[0], 1)
        predictions.append(predicted)
        log_prob += torch.log(orig_probs.squeeze()[predicted[0].item()]).item()

        decoder_input = predicted
        for b in range(bsz):
            if not done[b]:
                if predicted[b][-1].item() == END_IDX:
                    done[b] = True
                    total_done += 1
        if total_done == bsz:
            break
    predictions = torch.cat(predictions, dim = -1)
    return predictions, log_prob

def mini_batchify(sentence, chat_dictionary, device):
    """
    function to mimic batch function on smaller scale
    """
    RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)
    sentence =  RETOK.findall(sentence)
    batch = {
        'text_vecs': torch.tensor([chat_dictionary.t2v(sentence)], dtype=torch.long, device=device),
        'text_lens': torch.tensor([len(sentence)], dtype=torch.long),
        'use_packed': True, 
    }
    return batch

def format_chatbot_output(reply):
    """
    function for making seq2seq chatbot output more human readable
    """
    i_am = "i ' m"
    contraction = "n ' t "
    possessive = " ' s "
    i_am = re.compile(i_am)
    contraction = re.compile(contraction)
    possessive = re.compile(possessive)
    if re.search(r'’', reply):
        reply = re.sub(r'’', "'", reply)
    if i_am.search(reply):
        reply = i_am.sub("i'm ", reply)
    if contraction.search(reply):
        reply = contraction.sub("n't ", reply)
    if possessive.search(reply):
        reply = possessive.sub("'s ", reply)
    if re.search(r' , ', reply):
        reply = re.sub(r' , ', ', ', reply)
    if re.search(r"\w ' ", reply):
        reply = re.sub(r"\w ' ", "' ", reply)
    if re.search(r' \? ', reply):
        reply = re.sub(r'\?', '? ', reply)
    return reply

def format_user_input(reply):
    if re.search(r"'", reply):
        reply = re.sub(r"'", " ' ", reply)
    if re.search(r'\?', reply):
        reply = re.sub(r'\?', ' ? ', reply)
    return reply

def bash_format_text(text, *args):
    formatting = ''
    for arg in args:
        formatting += BASH_FORMATTING[arg]
    return formatting + text + BASH_FORMATTING['END']

def start_rapbot(model, chat_dictionary, p, device, transformer = False):
    """
    for chatting with seq2seq chatbot
    """
    input_sentence = input('User > ')
    input_sentence = input_sentence.lower()
    input_sentence = format_user_input(input_sentence)

    continue_convo = True
    user_batch = mini_batchify(input_sentence, chat_dictionary, device)

    while continue_convo:
        context = chat_dictionary.v2t(user_batch['text_vecs'][0].tolist())
        decoded_sent, _ = nucleus_sample(model, user_batch, chat_dictionary, p, device, False, transformer)
        bot_reply = decoded_sent[0][1:-1]
        bot_reply_readable = chat_dictionary.v2t(bot_reply.tolist())
        bot_reply_readable = format_chatbot_output(bot_reply_readable)
        print(BASH_FORMATTING['YELLOW'] + BASH_FORMATTING['BOLD']  + 'Aubrey: {}'.format(bot_reply_readable) + BASH_FORMATTING['END'])
        context += '\n ' + bot_reply_readable
        response = input('User > ')
        if (response == 'q' or response == 'quit' or response == 'exit'):
            continue_convo = False
        context += '\n ' + response
        user_batch = mini_batchify(context, chat_dictionary, device)
    return context

def transfer_learning_bot(model, tokenizer, max_length, top_k, top_p):
    """
    for chatbot trained using transfer learning
    """
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
