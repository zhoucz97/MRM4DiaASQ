#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

import pickle
from evaluate import evaluate
from models.losses import log_likelihood
from utils.dataset import CustomDataset
from models.collate import collate_fn, gold_labels
import numpy as np
from models.model import Ranker
from utils.processor import QuadDataProcessor
from utils.tager import SpanTagging, TableTagging, SpanTypeTagging
from utils.util import set_seed, set_labels
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


def do_test(args, mode_type):
    set_seed(args.seed)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    # create processor
    processor = QuadDataProcessor(tokenizer, args.max_seq_len, args.span_max_len)
    
    print("Loading {} Dataset...".format(mode_type))
    if mode_type == 'train':
        test_dataset = CustomDataset('train', os.path.join(args.data_dir, args.train_path), processor, tokenizer, args.max_seq_len)
    elif mode_type == 'test':
        test_dataset = CustomDataset('test', os.path.join(args.data_dir, args.test_path), processor, tokenizer, args.max_seq_len)
    elif mode_type == 'valid':
        test_dataset = CustomDataset('valid', os.path.join(args.data_dir, args.valid_path), processor, tokenizer, args.max_seq_len)
    else:
        assert 1 == 2
    print("Construct Dataloader...")
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    print("Building SPAN-ASTE model...")
    # get dimension of target and relation
    span_nclass, span_type_nclass, table_nclass = len(SpanTagging), len(SpanTypeTagging), len(TableTagging)
    # build span-aste model
    model = Ranker(
        args.bert_model,
        span_nclass,
        span_type_nclass,
        table_nclass,
        device=args.device
    )

    if args.ckpt_path != 'none':
        model_path = args.ckpt_path
    else:
        assert 1 == 2
    model = torch.load(model_path, map_location=torch.device(args.device))
    model.eval()

    macro_res, topk_sum, all_sum, rank_res = evaluate(model, test_dataloader, args)
    if args.save_rank:
        if not os.path.exists(args.mode_save_path):
            os.mkdir(args.mode_save_path)
        save_file_name = str(args.lan)+'.'+str(args.top_k)+'.'+str(mode_type)+'.pkl'
        with open(os.path.join(args.mode_save_path, save_file_name), 'wb') as file:
            pickle.dump(rank_res, file)

    print(macro_res)
    print('topk num / all num =', topk_sum, '/', all_sum)
    return None


def format_print(name, p, r, f1):
    strs = "{name:<20}P:{p:<10.2f}R:{r:<10.2f}F1:{f1:<10.2f}".format(name=name, p=p, r=r, f1=f1)
    print(strs)


def do_train(args):
    # set seed
    set_seed(args.seed)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    # create processor
    processor = QuadDataProcessor(tokenizer, args.max_seq_len, args.span_max_len)

    print("Loading Train & Valid & Test Dataset...")
    # eval_dataset = CustomDataset("test", args.valid_path, processor, tokenizer, args.max_seq_len)
    eval_dataset = CustomDataset("dev", os.path.join(args.data_dir, args.valid_path), processor, tokenizer, args.max_seq_len)
    test_dataset = CustomDataset('test', os.path.join(args.data_dir, args.test_path), processor, tokenizer, args.max_seq_len)
    train_dataset = CustomDataset("train", os.path.join(args.data_dir, args.train_path), processor, tokenizer, args.max_seq_len)    
 
    print("Construct Dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("Building SPAN-RANKER model...")
    # get dimension of target and relation
    span_nclass, span_type_nclass, table_nclass = len(SpanTagging), len(SpanTypeTagging), len(TableTagging)
    # build span-aste model
    model = Ranker(
        args.bert_model,
        span_nclass,
        span_type_nclass,
        table_nclass,
        device=args.device
    )
    model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    bert_params = list(model.bert.named_parameters())
    other_params = list(model.span_representation.named_parameters()) + list(model.span_ffnn.named_parameters()) \
         + list(model.span_type_ffnn.named_parameters())
    optimizer_grouped_params = [
        {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 
         'weight_decay': args.weight_decay,
         'lr': args.plm_lr},
        {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0,
         'lr': args.plm_lr},
        {'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)], 
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in other_params if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    
    print("Building Optimizer...")
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=args.learning_rate)
    
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    best_f1 = 0
    best_epoch = 0
    loss_list = []
    for epoch in range(1, args.num_epochs + 1):
        for batch_ix, batch in enumerate(tqdm(train_dataloader)):

            guid, tok2id, id2tok, input_ids, span_indices, span_labels, span_type_labels, span_postag_labels = batch
            input_ids = torch.LongTensor(input_ids).to(args.device)
            # forward
            model.train()
            span_probs, span_type_probs = model(input_ids, span_indices, span_postag_labels)
            
            loss_ner = log_likelihood(span_probs, span_labels, args.device)
            loss_ner2 = log_likelihood(span_type_probs, span_type_labels, args.device)

            loss = loss_ner2
            loss.backward()
            
            if batch_ix % args.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            loss_list.append(float(loss))

        macro_res, topk_sum, all_sum, _ = evaluate(model, eval_dataloader, args)
        print('dev result:')
        print(macro_res)

        macro_f1 = macro_res['macro avg']['f1-score']
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model_path = os.path.join(args.output_dir, 'model_best.' + args.lan + '.pt')
            torch.save(model, model_path)
            test_macro_res, topk_sum, all_sum, _ = evaluate(model, test_dataloader, args)
            print('Test result:')
            print(test_macro_res)

    print('best epoch: {}\tbest dev f1: {:.4f}\n\n'.format(best_epoch, best_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="the path of plm.")
    parser.add_argument("--train_path", default="data/15res", type=str, help="The path of train set.")
    parser.add_argument("--valid_path", default="data/15res", type=str, help="The path of dev set.")
    parser.add_argument("--test_path", default="data/15res", type=str, help="The path of test set.")
    parser.add_argument("--output_dir", default='outputs', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--data_dir", default='../../data', type=str,
                        help="The basic data dir.")
    parser.add_argument("--ckpt_path", default=None, type=str,
                        help="The path of model parameters for initialization.")
    
    # hyper-parameters
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for testing/eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--plm_lr", default=1e-5, type=float, help="The initial learning rate for plm.")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--grad_accumulation_steps", default=2, type=int, help="")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=30, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=50, type=int,
                        help="The interval steps to evaluate model performance.")
    parser.add_argument("--top_k", default=512, type=int, help="top k")
    parser.add_argument("--span_max_len", default=10, type=int, help="the max len of span.")
    
    # other params
    parser.add_argument("--init_time", default=None, type=str,
                        help="the initial time of running codes.")
    parser.add_argument("--mode", default=None, type=str,
                        help="train or test")
    parser.add_argument('--seed_list', nargs='+', type=int, default=[125, 126, 127, 128, 129])
    parser.add_argument("--mode_type_list", nargs='+', type=str, default=['test'],
                        help="the type of testing. train/test/valid")
    parser.add_argument("--mode_save_path", default=None, type=str,
                        help="the save_path of ranking file.")
    parser.add_argument("--device", default=None, type=str,
                        help="cuda or cpu")
    parser.add_argument("--lan", default='en', type=str,
                        help="en")
    
    parser.add_argument("--save_rank", default=False, type=bool)
    

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"using device:{device}")
    args.device = device
    
    
    if args.mode == 'train':
        do_train(args)
    elif args.mode == 'test':
        for mode_type in args.mode_type_list:
            do_test(args, mode_type)
    else:
        print("Error! Please specify the mode to be train or test!")
        exit(0)
