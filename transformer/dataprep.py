import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


special_symbols = [-1,-2,-3]
BOS_IDX = 0
PAD_IDX = 1
EOS_IDX = 2
SRC_PAD_IDX = 999



class VocabTransform:
    def __init__(self, vocab_transform_dict):
        self.vocab_transform_dict = vocab_transform_dict

    def __call__(self, raw_input_list):
        return list(map(self.vocab_transform_dict.get, raw_input_list))

def create_vocab(vocab_transform, special_symbols, max_size=45):
    count = 0
    offset = len(special_symbols)
    for i in special_symbols:
        vocab_transform[i] = count
        count += 1
    for i in np.arange(0, max_size, 0.5):
        vocab_transform[i] = count
        count += 1
    return vocab_transform

def sequetial_transforms(*transforms):
    def func(inp):
        for transform in transforms:
            inp = transform(inp)
        return inp
    return func

def tensor_transform(token_ids):
    return torch.cat((
        torch.tensor([BOS_IDX]),
        torch.tensor(token_ids),
        torch.tensor([EOS_IDX])
    ))

def tensor_transform_cnn(token_ids):
    return torch.tensor(token_ids)

def collate_fn(seq_transform, batch_first=False):
    def func(batch):
        tgt_batch = []
        src_batch = []
        for src, tgt in batch:
            tgt_batch.append(seq_transform(tgt))
            src_batch.append(src)

        src_batch = pad_sequence(src_batch, padding_value=SRC_PAD_IDX, batch_first=batch_first)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=batch_first)
        return src_batch, tgt_batch
    return func


def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask==0, float("-inf")).masked_fill(mask==1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool) 

    src_padding_mask = (src==SRC_PAD_IDX).transpose(0,1) 
    src_padding_mask = src_padding_mask[:,:,0].squeeze(-1)

    tgt_padding_mask = (tgt==PAD_IDX).transpose(0,1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

        