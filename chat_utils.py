import re
import torch.nn.functional as F
from training_utils import *

# T0-D0
# include Beam() class?
# include nucleus sampling functions


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
    RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)
    sentence =  RETOK.findall(sentence)
    batch = {
        'text_vecs': torch.tensor([chat_dictionary.t2v(sentence)], dtype=torch.long, device=device),
        'text_lens': torch.tensor([len(sentence)], dtype=torch.long),
        'use_packed': True, 
    }
    return batch

def start_rapbot(model, chat_dictionary, p, device, transformer = False):
    input_sent = input('User > ')
    input_sent = input_sent.lower()

    chat_log = {}
    continue_convo = True
    user_batch = mini_batchify(input_sent, chat_dictionary, device)

    while continue_convo:
        context = chat_dictionary.v2t(user_batch['text_vecs'][0].tolist())

        decoded_sent, _ = nucleus_sample(model, user_batch, chat_dictionary, p, device, False, transformer)

        bot_reply = decoded_sent[0][1:-1]
        bot_reply_readable = chat_dictionary.v2t(bot_reply.tolist())
        print('Chatty boy: {}'.format(bot_reply_readable))
        
        context += '\n ' + bot_reply_readable

        response = input('User > ')
        if (response == 'q' or response == 'quit'):
            break

        context += '\n ' + response

        user_batch = mini_batchify(context, chat_dictionary, device)
    return context
