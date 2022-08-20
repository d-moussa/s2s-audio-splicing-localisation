from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer
import math
from NEW.nn.transformer.dataprep import create_mask
import pathlib

import sys
sys.path.insert(0, "")

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        embs = self.embedding(tokens.long())
        # Signal verstärkung
        embs = embs * math.sqrt(self.emb_size)
        return embs

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos*den)
        if emb_size % 2 == 1:
            den = den[:-1]
        pos_embedding[:, 1::2] = torch.cos(pos*den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        addedEmbedding = token_embedding + self.pos_embedding[:token_embedding.size(0), :]
        return self.dropout(addedEmbedding)

class Seq2SeqTransformerMultiinput(nn.Module):
    def __init__(self,
                enc_args_list,
                memory_transformation,
                num_decoder_layers,
                emb_size,
                nhead,
                tgt_vocab_size,
                dim_feedforward=512,
                dropout=0.1):
        super(Seq2SeqTransformerMultiinput, self).__init__()

        self.enc_args_list = enc_args_list
        self.memory_transformation = memory_transformation
        #memory_layer_args # falls wir das wirklich brauchen?

        self.encoder_list = []
        self.input_total_size = 0
        for enc_args in self.enc_args_list:
            encoder_layers = TransformerEncoderLayer(
                d_model=enc_args["d_model"], 
                nhead=enc_args["nhead"], 
                dim_feedforward=enc_args["dim_feedforward"]
                )
            self.input_total_size += enc_args["d_model"]
                
            self.encoder_list.append(TransformerEncoder(encoder_layers, num_layers=enc_args["num_layers"]))
        
        self.encoder_list = torch.nn.ModuleList(self.encoder_list)

        if memory_transformation == "projection":
            self.memory_processing = torch.nn.Linear(self.input_total_size, emb_size)

        if memory_transformation == "attention":
            self.attn_weight_layer = torch.nn.GRU(self.input_total_size, len(self.enc_args_list), 1)
            self.attn_weight_softmax = torch.nn.Softmax(dim=1)

        
        decoder_layers = TransformerDecoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_token_embedding = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding_tgt = PositionalEncoding(emb_size, dropout=dropout)
        self.positional_encoding_src = PositionalEncoding(self.input_total_size, dropout=dropout)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding_src(src)
        tgt_emb = self.positional_encoding_tgt(self.tgt_token_embedding(tgt))
        
        #outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        start_pos = 0
        mem_parts = []
        for i in range(len(self.enc_args_list)):
            end_pos = self.enc_args_list[i]["d_model"] + start_pos
            src_emb_part = src_emb[:,:,start_pos:end_pos]
            memory_part = self.encoder_list[i](src_emb_part, mask=src_mask, src_key_padding_mask=src_padding_mask)
            mem_parts.append(memory_part)
            start_pos = end_pos
        # TODO hier über alle encoder iterieren und via concat, attention, projection output generieren
        
        memory = torch.cat(mem_parts, dim=-1)

        if self.memory_transformation == "projection":
            ret = self.memory_processing(memory)

        elif self.memory_transformation == "attention":
            mem_weight, _ = self.attn_weight_layer(memory)
            mem_weight = self.attn_weight_softmax(mem_weight[-1, :])
            mem_weight = torch.unsqueeze(mem_weight, 1)

            start_pos = 0
            ret_parts = []
            for i in range(len(self.enc_args_list)):
                end_pos = self.enc_args_list[i]["d_model"] + start_pos
                #memory[:,:,start_pos:end_pos] = memory[:,:,start_pos:end_pos] * mem_weight[:,i]
                ret_parts.append(torch.mul(memory[:,:,start_pos:end_pos], mem_weight[:, :, i]))
                start_pos = end_pos
            ret = torch.cat(ret_parts, dim=-1)
        else:
            ret = memory

        outs = self.decoder(tgt_emb, ret, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        
        return self.generator(outs)

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding_src(src)
        start_pos = 0
        mem_parts = []
        for i in range(len(self.enc_args_list)):
            end_pos = self.enc_args_list[i]["d_model"] + start_pos
            src_emb_part = src_emb[:,:,start_pos:end_pos]
            memory_part = self.encoder_list[i](src_emb_part, mask=src_mask)
            mem_parts.append(memory_part)
            start_pos = end_pos
        # TODO hier über alle encoder iterieren und via concat, attention, projection output generieren
        
        memory = torch.cat(mem_parts, dim=-1)

        if self.memory_transformation == "projection":
            memory = self.memory_processing(memory)

        elif self.memory_transformation == "attention":
            mem_weight, _ = self.attn_weight_layer(memory)
            mem_weight = self.attn_weight_softmax(mem_weight[-1, :])
            mem_weight = torch.unsqueeze(mem_weight, 1)

            start_pos = 0
            ret_parts = []
            for i in range(len(self.enc_args_list)):
                end_pos = self.enc_args_list[i]["d_model"] + start_pos
                #memory[:,:,start_pos:end_pos] = memory[:,:,start_pos:end_pos] * mem_weight[:,i]
                ret_parts.append(torch.mul(memory[:,:,start_pos:end_pos], mem_weight[:, :, i]))
                start_pos = end_pos
            ret = torch.cat(ret_parts, dim=-1)

        return memory

    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.tgt_token_embedding(tgt)
        tgt_emb = self.positional_encoding_tgt(tgt_emb)
        return self.decoder(tgt_emb, memory, tgt_mask)

def feed_batch(src, tgt, model, loss_fn, optimizer, device, usecase="Train"):
    src = src.to(device)
    tgt = tgt.to(device)

    tgt_input = tgt[:-1, :]
    src_mask, tgt_mask,src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

    tgt_out = tgt[1:,:]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    if usecase == "Train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss

def train_epoch(model, optimizer, train_dl, val_dl, loss_fn, device, subset=1.0):
    model.train()
    train_losses = 0
    eval_losses = 0

    for uc, dl in zip(["Train", "Eval"], [train_dl, val_dl]):
        batch_num = 0
        total_batches = len(dl)
        subset_break_point = int(total_batches*subset)
        if uc == "Train":
            for src, tgt in dl:
                loss = feed_batch(src, tgt.long(), model, loss_fn, optimizer, device, usecase=uc)
                train_losses += loss.item()
                print("TRAINING - Batch: "+str(batch_num)+"/"+str(total_batches)+", Loss: "+str(loss.item()), end='\r')
                batch_num += 1
                if subset_break_point <= batch_num:
                    break
        elif uc == "Eval":
            with torch.no_grad():
                model.eval()
                for src, tgt in dl:
                    loss = feed_batch(src, tgt, model, loss_fn, optimizer=None, device=device, usecase=uc)
                    eval_losses += loss.item()
                    print("EVALUATION - Batch: "+str(batch_num)+"/"+str(total_batches)+", Loss: "+str(loss.item()), end='\r')
                    batch_num += 1

    return train_losses/len(train_dl), eval_losses/len(val_dl)

def save_checkpoint(checkpoint_dir, model, optimizer, misc = None):
    p = pathlib.Path(checkpoint_dir)
    p_dir = p.parents[0]
    pathlib.Path(p_dir).mkdir(parents=True, exist_ok=True)
    if misc is not None and type(misc) is dict:
        save_dict = misc
        save_dict["model_state_dict"] = model.state_dict()
        save_dict["optimizer_state_dict"] = optimizer.state_dict()
    else:
        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
    torch.save(save_dict, checkpoint_dir)
    print("Model and optimizer saved!")

def load_checkpoint(checkpoint_dir, model, optimizer, device, misc_keys=None):
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except:
        print("Unable to load the optimizer!")
    misc_values = {}
    if misc_keys is not None:
        for m in misc_keys:
            misc_values[m] = checkpoint[m]
    model.to(device)
    print("Model and optimizer loaded!")
    return model, optimizer, misc_values