#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from evaluate import evaluate
from models.losses import log_likelihood, loss_func
from models.metrics import SpanEvaluator, QuadEvaluator
from utils.bar import ProgressBar
from utils.dataset import CustomDataset
from models.collate import collate_fn, gold_labels
import numpy as np
from models.model import Model
from utils.processor import QuadDataProcessor
from utils.tager import SpanTagging, TableTagging, SpanTypeTagging
from utils.util import set_seed, set_labels
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup, 
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"using device:{device}")


metric_name_list = ['targets', 'aspects', 'opinions', 'ta', 'to', 'ao', 'quad', 'iden', 'intra', 'inter']


def format_print(name, p=None, r=None, f1=None):
    if p != None and r != None:
        strs = "{name:<20} | P:{p:<5.2f} | R:{r:<5.2f} | F1:{f1:<5.2f}".format(name=name, p=p, r=r, f1=f1)
    else:
        strs = "{name:<20} | F1:{f1:<5.2f}".format(name=name, f1=f1)
    print(strs)
    

def do_test(args, seed):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.plm_dir, args.plm_name))
    # create processor
    processor = QuadDataProcessor(tokenizer, args.max_seq_len, args)

    print("Loading Test Dataset...")
    test_dataset = CustomDataset('test', args.test_path, processor, tokenizer, args.max_seq_len)
    print("Construct Dataloader...")
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    print("Building MRM model...")
    # get dimension of target and relation
    span_nclass, span_type_nclass, table_nclass = len(SpanTagging), len(SpanTypeTagging), len(TableTagging)
    # build span-aste model
    model = Model(
        os.path.join(args.plm_dir, args.plm_name),
        span_nclass,
        span_type_nclass,
        table_nclass,
        device=device,
        args=args
    )
    
    if args.mode == 'test' and args.ckpt_path != 'none':
        model_path = os.path.join(args.ckpt_path, args.plm_name+'.k'+str(args.top_k)+'.seed'+str(seed)+'.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError('No such ckpt', model_path)
    model = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.eval()

    span_scores, scores, _, _ = evaluate(args.test_path, model, test_dataloader, device)
    for i, name in enumerate(metric_name_list):
        format_print('Test ' + str(name), *scores[i])
    return scores


def do_train(args, seed):
    # set seed
    set_seed(seed)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.plm_dir, args.plm_name))
    # create processor
    processor = QuadDataProcessor(tokenizer, args.max_seq_len, args)

    print("Loading Train & Valid & Test Dataset...")
    # eval_dataset = CustomDataset("test", args.dev_path, processor, tokenizer, args.max_seq_len)
    eval_dataset = CustomDataset("valid", args.valid_path, processor, tokenizer, args.max_seq_len)
    test_dataset = CustomDataset('test', args.test_path, processor, tokenizer, args.max_seq_len)  
    train_dataset = CustomDataset("train", args.train_path, processor, tokenizer, args.max_seq_len)
    
    print("Construct Dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("Building MRM model...")
    # get dimension of target and relation
    span_nclass, span_type_nclass, table_nclass = len(SpanTagging), len(SpanTypeTagging), len(TableTagging)
    # build mrm model
    model = Model(
        os.path.join(args.plm_dir, args.plm_name),
        span_nclass,
        span_type_nclass,
        table_nclass,
        device=device,
        args=args
    )
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    bert_params = list(model.bert.named_parameters())
    other_params = list(model.span_representation.named_parameters()) + list(model.table_ffnn.named_parameters()) + \
        list(model.self_attn_reply.named_parameters()) + list(model.self_attn_speaker.named_parameters()) + \
            list(model.span_mapping.named_parameters()) + list(model.pair_representation.named_parameters())
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
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=args.lr)
    
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    best_epoch = 0
    best_scores = None
    best_test_scores = None
    best_valid_res, best_test_res = None, None
    for epoch in range(1, args.num_epochs + 1):
        loss_list = []
        print('==============Epoch: {} =============='.format(epoch))
        for batch_ix, batch in enumerate(tqdm(train_dataloader, mininterval=2)):

            guid, tok2id, id2tok, input_ids, attn_masks, span_indices, span_labels, span_type_labels, span_postag_labels, quads, \
                ranker_result, topk_span_indices_idx, table_labels, _, mask_rep, mask_spe, entities, sentence_idx_list = batch

            # forward
            model.train()
            span_pair_probs, _ = model(input_ids, attn_masks, span_indices, span_postag_labels, span_labels, topk_span_indices_idx, 
                                       mask_rep, mask_spe)

            bs, span_num, _, _ = span_pair_probs.size()

            loss_table = log_likelihood(span_pair_probs, table_labels, device, table_nclass)
            # loss_span = log_likelihood(span_type_probs, span_type_labels, device, span_type_nclass)
            
            loss = loss_table
            loss.backward()
            loss_list.append(float(loss))

            if batch_ix % args.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
        print('Epoch: {}, Average Loss: {:.4f}'.format(epoch, np.mean(loss_list)))
        
        if epoch < 25:
            continue
        
        valid_micro_score, _, valid_res = evaluate(args.valid_path, model, eval_dataloader, device)
        format_print('Valid TAOS Quad', *valid_micro_score)
        

        if best_scores == None or valid_micro_score[2] > best_scores[2]:
            best_scores = valid_micro_score
            best_valid_res = valid_res
            best_epoch = epoch
            test_micro_score, _, test_res = evaluate(args.test_path, model, test_dataloader, device)
            best_test_scores = test_micro_score
            best_test_res = test_res
            format_print('Test TAOS Quad', *test_micro_score)
            
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model_save_name = 'roberta-large' + '.k' + str(args.top_k) + '.seed' + str(seed) + '.pt'
            model_path = os.path.join(args.output_dir, model_save_name)
            torch.save(model, model_path)
    
    print('\nBest Epoch: ', best_epoch)
    print('Best Valid Scores:')
    for i, name in enumerate(metric_name_list):
        format_print('Valid '+name, *best_valid_res[name])
    print('Corresponding Test Scores:')
    for i, name in enumerate(metric_name_list):
        format_print('Test '+name, *best_test_res[name])
    # print(best_test_res)
    return best_test_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--plm_dir", default="", type=str,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--plm_name", default="bert-base-uncased", type=str,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for testing/eval.")
    parser.add_argument("--lr", default=1e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--plm_lr", default=1e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--grad_accumulation_steps", default=2, type=int, help="")
    parser.add_argument("--train_path", default="data/15res", type=str, help="The path of train set.")
    parser.add_argument("--valid_path", default="data/15res", type=str, help="The path of valid set.")
    parser.add_argument("--test_path", default="data/15res", type=str, help="The path of test set.")
    parser.add_argument("--output_dir", default='checkpoint/', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--seed_list', nargs='+', type=int, default=[125, 126, 127, 128, 129])
    parser.add_argument("--logging_steps", default=30, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=50, type=int,
                        help="The interval steps to evaluate model performance.")
    parser.add_argument("--ckpt_path", default=None, type=str,
                        help="The path of model parameters for initialization.")
    parser.add_argument("--init_time", default=None, type=str,
                        help="init time.")
    parser.add_argument("--mode", default=None, type=str,
                        help="mode")
    parser.add_argument("--ranker_path", default=None, type=str,
                        help="")
    parser.add_argument("--top_k", default=512, type=int,
                        help="")
    parser.add_argument("--cache_path", default=None, type=str,
                        help="")
    parser.add_argument("--lan", default=None, type=str,
                        help="dataset langugae")

    args = parser.parse_args()
    
    if args.mode == 'train':
        test_res = {}
        for i, name in enumerate(metric_name_list):
            test_res[name] = [[], [], []]
            
        for seed in args.seed_list:
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            best_test_res = do_train(args, seed)            
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('Training start & end time: ', start_time, end_time)
            for i, name in enumerate(metric_name_list):
                test_res[name][0].append(best_test_res[name][0])
                test_res[name][1].append(best_test_res[name][1])
                test_res[name][2].append(best_test_res[name][2])
        print('===== Average Test Result using Seed List {} ====='.format(str(args.seed_list)))
        for k, v in test_res.items():
            format_print(k, p=np.mean(v[0]), r=np.mean(v[1]), f1=np.mean(v[2]))
        
    elif args.mode == 'test':
        test_res = {}
        for i, name in enumerate(metric_name_list):
            test_res[name] = [[], [], []]
            
        for seed in args.seed_list:
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            best_test_socres = do_test(args, seed)            
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('Training start & end time: ', start_time, end_time)
            for i, name in enumerate(metric_name_list):
                test_res[name][0].append(best_test_socres[i][0])
                test_res[name][1].append(best_test_socres[i][1])
                test_res[name][2].append(best_test_socres[i][2])
        print('===== Average Test Result using Seed List {} ====='.format(str(args.seed_list)))
        for k, v in test_res.items():
            format_print(k, p=np.mean(v[0]), r=np.mean(v[1]), f1=np.mean(v[2]))
        
    else:
        print("Error! Please specify the mode to be train or test!")
        exit(0)
