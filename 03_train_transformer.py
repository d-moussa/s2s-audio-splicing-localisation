import sys
sys.path.insert(0, "")

import torch
import transformer.dataprep as dataprep

import gc
import json

from transformer.dataprep import VocabTransform
from transformer.dataprep import create_vocab, sequetial_transforms, tensor_transform, special_symbols, collate_fn
from transformer.dataset import PickleDataset, PickleDatasetMultiinput

import transformer.s2s_transformer as transformer
import transformer.s2s_evaluation as evaluator
from transformer.s2s_transformer import Seq2SeqTransformer
from transformer.s2s_transformer_m import Seq2SeqTransformerMultiinput
from timeit import default_timer as timer
import os
from os.path import join, isfile
import re
from pathlib import Path

import configs.data.config_ace_clean_01 as config_ace_clean_01

import configs.model.ace_single_clean_transformer_m as ace_single_clean_transformer_m

def create_dirs(m_config, d_config):
    Path(m_config.save_model_dir).mkdir(parents=True, exist_ok=True)

def create_pickle_path_dicts(pickle_path, name):
    retVal = {}
    onlyfiles = [f for f in os.listdir(pickle_path) if isfile(join(pickle_path, f))]
    filtered_files = [f for f in onlyfiles if (name in f) and (f.endswith(".pkl"))]
    if not name.endswith("_m"):
        filtered_files = [f for f in filtered_files if ("_m_" not in f)]
    nums = [int(re.findall(r'\d+', f)[-1]) for f in filtered_files]
    for k, v in zip(nums, filtered_files):
        retVal[k] = pickle_path+v
    return retVal

def create_dataloaders(splice_seq_transform, d_config, m_config):
    train_pickle_dict = create_pickle_path_dicts(d_config.dataset_root_path, d_config.dataset_pickle_name_train)
    eval_pickle_dict = create_pickle_path_dicts(d_config.dataset_root_path, d_config.dataset_pickle_name_val)
    test_pickle_dict = create_pickle_path_dicts(d_config.dataset_root_path, d_config.dataset_pickle_name_test)

    if d_config.input_specs_types is None:
        ds_train = PickleDataset(train_pickle_dict, d_config.dataset_path_train, d_config.package_size)
        ds_eval = PickleDataset(eval_pickle_dict, d_config.dataset_path_val, d_config.package_size)
        ds_test = PickleDataset(test_pickle_dict, d_config.dataset_path_test, d_config.package_size)
    else:
        ds_train = PickleDatasetMultiinput(train_pickle_dict, d_config.dataset_path_train, d_config.package_size)
        ds_eval = PickleDatasetMultiinput(eval_pickle_dict, d_config.dataset_path_val, d_config.package_size)
        ds_test = PickleDatasetMultiinput(test_pickle_dict, d_config.dataset_path_test, d_config.package_size)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=m_config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn(splice_seq_transform))
    dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=m_config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn(splice_seq_transform))
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn(splice_seq_transform))
    return dl_train, dl_eval, dl_test

def create_model(vocab_size, m_config, d_config, device):

    if m_config.use_single_encoder:
        seq2seq_transformer = Seq2SeqTransformer(
                m_config.num_encoder_layers,
                m_config.num_decoder_layers,
                m_config.emb_size,
                m_config.nhead,
                vocab_size,
                m_config.ffn_hid_dim
            )
    else:
        seq2seq_transformer = Seq2SeqTransformerMultiinput(
            m_config.enc_args_list,
            m_config.encoder_memory_transformation,
            m_config.num_decoder_layers,
            m_config.emb_size,
            m_config.nhead,
            vocab_size,
            m_config.ffn_hid_dim
        )

    for p in seq2seq_transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform(p)
    seq2seq_transformer = seq2seq_transformer.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataprep.PAD_IDX)
    optimizer = torch.optim.Adam(seq2seq_transformer.parameters(), lr=m_config.lr, betas=m_config.betas, eps=m_config.eps)
    return seq2seq_transformer, loss_fn, optimizer

def train_transformer(seq2seq_transformer, dl_train, dl_val, optimizer, loss_fn, m_config, d_config, device, subset=1.0):
    train_losses = []
    eval_losses = []
    loss_diff = 0
    eval_loss_lastmin = 1000
    loss_increase_count = 0
    best_val_loss = 10000

    for epoch in range(1, m_config.num_epochs+1):
        start_time = timer()
        train_loss, eval_loss = transformer.train_epoch(seq2seq_transformer, optimizer, dl_train, dl_val, loss_fn, device, subset)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        end_time = timer()
        print("Epoch: ",epoch, " Train Loss: ", train_loss, " Val Loss: ",eval_loss, " Epoc Time: ",(end_time- start_time))

        if (eval_loss+m_config.early_stopping_delta) >= best_val_loss:
            loss_increase_count += 1
            print("loss increase count: ",loss_increase_count)
            if loss_increase_count == m_config.early_stopping_wait:
                break
        else:
            loss_increase_count = 0
    
        checkpoint_model_name = m_config.train_set_name+"_"+m_config.model_name + "_" + m_config.experiment_name+"_"+m_config.splice_name+".pth"
        checkpoint_save_path = m_config.save_model_dir+checkpoint_model_name
        
        if eval_loss < best_val_loss:
            print("Model improved: ", eval_loss," | ",best_val_loss)
            transformer.save_checkpoint(checkpoint_save_path, seq2seq_transformer, optimizer)
            best_val_loss = eval_loss
            
    return train_losses, eval_losses

def train(m_config, d_train_config):
    create_dirs(m_config, d_train_config)

    print(" --- TRAINING: ", d_train_config.config_name, m_config.model_name, " ---")
    device = "cuda:0"
    subset = 1

    vocab_transform_dict = {}
    vocab_transform_dict = create_vocab(vocab_transform_dict, special_symbols)
    vocab_transform = VocabTransform(vocab_transform_dict)

    splice_seq_transform = sequetial_transforms(vocab_transform, tensor_transform)
    vocab_size = len(vocab_transform_dict)
    seq2seq_transformer, loss_fn, optimizer = create_model(vocab_size, m_config, d_train_config, device)

    if m_config.load_checkpoint_path is not None:
        print("LOADING CHECKPOINT")
        seq2seq_transformer, optimizer, _ = transformer.load_checkpoint(m_config.load_checkpoint_path, seq2seq_transformer, optimizer, device)

    dl_train, dl_eval, dl_test = create_dataloaders(splice_seq_transform, d_train_config, m_config)
    train_losses, eval_losses = train_transformer(seq2seq_transformer, dl_train, dl_eval, optimizer, loss_fn, m_config, d_train_config, device, subset=subset)  

def load_predict(m_config, d_test_config):

    print(" --- PREDICTIONS: ", d_test_config.config_name, m_config.model_name, " ---")

    device = "cuda:0"
    vocab_transform_dict = {}
    vocab_transform_dict = create_vocab(vocab_transform_dict, special_symbols)
    vocab_transform = VocabTransform(vocab_transform_dict)
    splice_seq_transform = sequetial_transforms(vocab_transform, tensor_transform)
    
    vocab_size = len(vocab_transform_dict)

    if m_config.use_single_encoder:
        loaded_seq2seq = Seq2SeqTransformer(
                m_config.num_encoder_layers,
                m_config.num_decoder_layers,
                m_config.emb_size,
                m_config.nhead,
                vocab_size,
                m_config.ffn_hid_dim
            )
    else:
        loaded_seq2seq = Seq2SeqTransformerMultiinput(
            m_config.enc_args_list,
            m_config.encoder_memory_transformation,
            m_config.num_decoder_layers,
            m_config.emb_size,
            m_config.nhead,
            vocab_size,
            m_config.ffn_hid_dim
        )
    
    loaded_optimizer = torch.optim.Adam(loaded_seq2seq.parameters(), lr=m_config.lr, betas=m_config.betas, eps=m_config.eps)

    checkpoint_model_name = m_config.train_set_name+"_"+m_config.model_name + "_" + m_config.experiment_name+"_"+m_config.splice_name+".pth"
    checkpoint_save_path = m_config.save_model_dir+checkpoint_model_name
    loaded_seq2seq, loaded_optimizer, _ = transformer.load_checkpoint(checkpoint_save_path, loaded_seq2seq, loaded_optimizer, device)

    print("Loaded: ",checkpoint_save_path)
    param_count = sum(p.numel() for p in loaded_seq2seq.parameters() if p.requires_grad)
    print("trainable parameters: ",param_count)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataprep.PAD_IDX)
    
    test_pickle_dict = create_pickle_path_dicts(d_test_config.dataset_root_path, d_test_config.dataset_pickle_name_test)
    
    if d_test_config.input_specs_types is None:
        ds_test = PickleDataset(test_pickle_dict, d_test_config.dataset_path_test, d_test_config.package_size)
    else:
        ds_test = PickleDatasetMultiinput(test_pickle_dict, d_test_config.dataset_path_test, d_test_config.package_size)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn(splice_seq_transform))
    
    pred_count = 1 if d_test_config.max_split_points == 1 else 5
    beam_widths = [1,3,5,10,20]
    
    if d_test_config.max_split_points == 1:
        beam_widths = [5]

    ret_dict = evaluator.evaluate_beam(loaded_seq2seq, dl_test, vocab_transform_dict, pred_count, beam_widths, device)

    result_name = m_config.train_set_name+"-"+m_config.experiment_name+"-"+d_test_config.set_name+"-"+d_test_config.experiment_name+"-"+m_config.model_name+"-"+str(d_test_config.min_split_points)+"_"+str(d_test_config.max_split_points)+".json"
    result_save_path = m_config.save_model_dir+result_name
    with open(result_save_path, "w") as fp:
        json.dump(ret_dict, fp)
    print("test done!")


if __name__ == '__main__':
    d_train_confs = [
        config_ace_clean_01
    ]

    d_test_confs = [
        config_ace_clean_01
    ]
    m_confs = [
        ace_single_clean_transformer_m
    ]

    #for d_train_conf, m_conf in zip(d_train_confs, m_confs):
    #    train(m_conf, d_train_conf)
    
    # Uncomment this for prediction generation
    for d_test_conf, m_conf in zip(d_test_confs, m_confs):
        load_predict(m_conf, d_test_conf)