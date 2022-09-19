from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformer.dataprep import BOS_IDX, EOS_IDX, create_mask, generate_square_subsequent_mask
from transformer.dataprep import special_symbols
import pathlib
from pathlib import Path
import json
from pytorch_beam_search import seq2seq
import torch.utils.data as tud

def evaluate(model, val_dl, loss_fn, vocab_transform_dict, m_config, d_config, device):
    model.eval()
    losses = 0
    ret_list = []

    vocab_reverse_transform_dict = {y:x for x,y in vocab_transform_dict.items()}
    
    i = 0
    ret_dict = {}
    for src, tgt in val_dl:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask,src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        real = tgt_out[:-1,:].cpu().numpy().reshape(-1,tgt_out.shape[0]-1)
        
        logits = torch.nn.functional.softmax(logits, dim=2)
        probs_k, wrds_k = torch.topk(logits, k=5, dim=2)
        preds_k = wrds_k[:-1,:].cpu().numpy().reshape(-1,5)
        probs_k = probs_k[:-1,:].detach().cpu().reshape(-1,5).numpy()

        
        topk_preds = {}
        topk_pred_probs = {}
        for w in range(len(preds_k)):
            topk_preds[w] = preds_k[w].reshape(-1).tolist()
            topk_preds[w] = list(map(vocab_reverse_transform_dict.get, topk_preds[w]))
            topk_pred_probs[w] = probs_k[w].reshape(-1).tolist()

        t_target = list(map(vocab_reverse_transform_dict.get, real.reshape(-1)))
        ret_dict[i] = {
            "predictions": topk_preds,
            "probabilities": topk_pred_probs,
            "real": t_target
        }
        i += 1
        if i % 1000 == 0:
            print(i)

    result_save_path = m_config.save_model_dir+d_config.save_model_subdir+d_config.save_results_name
    with open(result_save_path, "w") as fp:
        json.dump(ret_dict, fp)
    return losses/len(val_dl), ret_dict

# TODO 
def greedy_decode(model, src, src_mask, max_len, topk=3, device="cpu"):
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.ones(1,1).fill_(BOS_IDX).type(torch.long).to(device) #special_symbols[0]
        wordsTopK = []
        probsTopK = []
        for i in range(max_len-1):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool))
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0,1)
            logits = model.generator(out[:,-1])
            logits = torch.nn.functional.softmax(logits)
            probs_k, wrds_k = torch.topk(logits, k=topk)

            wordsTopK.append(wrds_k)
            probsTopK.append(probs_k)

            next_word = wrds_k[0][0]
            if (wrds_k[0][0] == EOS_IDX) and (probs_k[0][0] <= 0.8):
                next_word = wrds_k[0][1] 

            ys = torch.cat([ys, torch.ones(1,1).type_as(src.data).fill_(next_word)], dim=0)
    return logits, ys, wordsTopK, probsTopK


def evaluate_greedy(model, val_dl, vocab_transform_dict, m_config, d_config, device):
    model.eval()
    ret_dict = {}

    vocab_reverse_transform_dict = {y:x for x,y in vocab_transform_dict.items()}
    
    i = 0
    ret_dict = {}
    for src, tgt in val_dl:
        src = src.to(device)
        src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool)
        src_mask = src_mask.to(device)
        logits, sentence, topk_words, topk_probs = greedy_decode(model, src, src_mask, d_config.max_split_points+2, topk=5, device=device)

        topk_preds = {}
        topk_pred_probs = {}
        for w in range(len(topk_words)):
            topk_preds[w] = topk_words[w].cpu().numpy().reshape(-1).tolist()
            topk_pred_probs[w] = topk_probs[w].cpu().numpy().reshape(-1).tolist()

        t_target = tgt[1:-1].cpu().numpy().reshape(-1).astype(int).tolist()
        ret_dict[i] = {
            "predictions": topk_preds,
            "probabilities": topk_pred_probs,
            "real": t_target
        }

        i += 1
        if i % 1000 == 0:
            print(i)
            break
    return ret_dict

def beam_search(model, 
                X,
                src_mask,
                predictions = 20,
                beam_width = 5,
                batch_size = 50, 
                progress_bar = 1):
    
    with torch.no_grad():
        samples_bs = X.size(1)
        device = next(model.parameters()).device
        memory = model.encode(X, src_mask)
        Y = torch.ones(1,samples_bs).fill_(BOS_IDX).type(torch.long).to(device)
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(Y.size(0), device).type(torch.bool))
        out = model.decode(Y, memory, tgt_mask)
        out = out.transpose(0,1)
        next_probabilities = model.generator(out[:,-1])

        vocabulary_size = next_probabilities.shape[-1]
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1)\
        .topk(k = beam_width, axis = -1)
        Y = Y.repeat((beam_width, 1))
        next_chars = next_chars.reshape(-1, samples_bs)
        Y = torch.cat((Y, next_chars), axis = -1)
        predictions_iterator = range(predictions - 1)
        for i in predictions_iterator:
            temp_x = X.repeat((1, beam_width, 1)).transpose(0, 1)
            dataset = tud.TensorDataset(temp_x, Y)
            loader = tud.DataLoader(dataset, batch_size = 256)
            next_probabilities = []
            iterator = iter(loader)
            for x, y in iterator:
                x = x.transpose(0, 1)
                y = y.transpose(0,1)
                memory = model.encode(x, src_mask)
                memory = memory.to(device)
                tgt_mask = (generate_square_subsequent_mask(y.size(0), device).type(torch.bool))
                out = model.decode(y, memory, tgt_mask)
                out = out.transpose(0,1)
                next_probs = model.generator(out[:,-1])
                probs = next_probs.log_softmax(-1)
                next_probabilities.append(probs)

            next_probabilities = torch.cat(next_probabilities, axis = 0)
            next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
            pr = probabilities.unsqueeze(-1)
            probabilities = torch.add(probabilities.unsqueeze(-1), next_probabilities)
            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(k = beam_width, axis = -1)
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(Y.shape[0] // beam_width, device = X.device).unsqueeze(-1) * beam_width
            Y = Y[best_candidates].flatten(end_dim = -2)
            Y = torch.cat((Y, next_chars), axis = 1)
        return Y.reshape(-1, beam_width, Y.shape[-1]), probabilities

def evaluate_beam(model, val_dl, vocab_transform_dict, predictions_count, beam_widths, device):
    model.eval()
    ret_dict = {}

    vocab_reverse_transform_dict = {y:x for x,y in vocab_transform_dict.items()}
    
    ret_dict = {}
    for beam_width in beam_widths:
        print("processing beam width ",str(beam_width))
        i = 0
        ret_dict["beam_"+str(beam_width)] = {}
        for src, tgt in val_dl:
            src = src.to(device)
            src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool)
            src_mask = src_mask.to(device)

            predictions, log_probabilities = beam_search(model, src, src_mask, batch_size=1, predictions=predictions_count, beam_width=beam_width)
            probs = torch.nn.functional.softmax(log_probabilities)

            predictions = predictions.cpu().numpy()
            predictions = predictions.reshape(predictions.shape[1], predictions.shape[2])
            probs = probs.cpu().numpy().reshape(-1)
            tgt = tgt.cpu().numpy()

            topk_preds = {}
            topk_pred_probs = {}
            for w in range(len(probs)):
                topk_preds[w] = predictions[w].reshape(-1).tolist() # skippen der BOS
                topk_pred_probs[w] = float(probs[w])
            ret_dict["beam_"+str(beam_width)][i] = {
                "predictions": topk_preds,
                "probabilities": topk_pred_probs,
                "real": tgt.reshape(-1).tolist()
            }
            i += 1
            if i % 2000 == 0:
                print(i, "/", len(val_dl))
    return ret_dict
