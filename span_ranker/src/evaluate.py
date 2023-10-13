#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import os
import random
from functools import partial

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.collate import gold_labels, collate_fn
from models.metrics import SpanEvaluator, QuadEvaluator
from models.model import Ranker
from utils.dataset import CustomDataset
from utils.processor import QuadDataProcessor
from utils.tager import SpanTagging, TableTagging
from utils.util import decode_quads, trans_str, cal_pair_score, cal_span_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pickle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate(model, data_loader, args):
    model.eval()
    pred_label_list, true_label_list = [], []
    topk_sum, all_sum = 0, 0
    k_sum = 0
    recalls = {
        'k': args.top_k,
        'span': [0, 0],
        'target': [0, 0],
        'aspect': [0, 0],
        'opinion': [0, 0]
    }
    save_dic = {}
    with torch.no_grad():
        for batch_ix, batch in enumerate(tqdm(data_loader)):
            guid, tok2id, id2tok, input_ids, span_indices, span_labels, span_type_labels, span_postag_labels = batch
            input_ids = torch.LongTensor(input_ids).to(args.device)
            conv_len = len(tok2id[0])
            
            _, span_type_probs = model(input_ids, span_indices, span_postag_labels)
            
            bs, span_num, n_class = span_type_probs.shape
            
            if args.top_k < 0:
                k = int(conv_len / (-args.top_k) )
                k = min(span_num, k)
            else:
                k = min(span_num, args.top_k)
            k_sum += k
                
            span_indices = torch.LongTensor(span_indices).to(args.device)
            # span_labels = torch.LongTensor(span_labels).to(args.device)
            span_type_labels = torch.LongTensor(span_type_labels).to(args.device)
            # topk_span_probs, topk_span_indices_idx = torch.topk(span_probs[:, :, 1], k, dim=-1)
            topk_span_type_probs, topk_span_type_indices_idx = torch.topk(1 - span_type_probs[:, :, 0], k, dim=-1)
            # topk_span_labels = span_labels.gather(dim=1, index=topk_span_indices_idx)
            topk_span_type_labels = span_type_labels.gather(dim=1, index=topk_span_type_indices_idx)
            
            dic = {'k': k, 'topk_span_indices_idx': topk_span_type_indices_idx.detach().cpu().tolist()}
            save_dic[guid[0]] = dic
            
            recalls['span'][0] += torch.sum(topk_span_type_labels!=0).item()
            recalls['span'][1] += torch.sum(span_type_labels!=0).item()
            recalls['target'][0] += torch.sum(topk_span_type_labels==1).item()
            recalls['target'][1] += torch.sum(span_type_labels==1).item()
            recalls['aspect'][0] += torch.sum(topk_span_type_labels==2).item()
            recalls['aspect'][1] += torch.sum(span_type_labels==2).item()
            recalls['opinion'][0] += torch.sum(topk_span_type_labels==3).item()
            recalls['opinion'][1] += torch.sum(span_type_labels==3).item()

            pred_span_type_labels = torch.argmax(span_type_probs.reshape(bs * span_num, n_class), dim=-1).cpu().tolist()
            span_type_labels = span_type_labels.reshape(-1).cpu().tolist()
            pred_label_list.extend(pred_span_type_labels)
            true_label_list.extend(span_type_labels)
    
    macro_res = classification_report(true_label_list, pred_label_list, digits=4, output_dict=True)
    
    # span_f1 = f1_score(true_label_list, pred_label_list)
    # span_pre = precision_score(true_label_list, pred_label_list)
    # span_rec = recall_score(true_label_list, pred_label_list)
    # span_scores = [span_pre, span_rec, span_f1]
    pair_scores = []
    print('sum_k: ', k_sum)
    print(recalls)
    return macro_res, topk_sum, all_sum, save_dic


def do_eval():
    set_seed(1024)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"using device:{device}")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # create processor
    processor = QuadDataProcessor(tokenizer, args.max_seq_len)

    print("Loading Train & Eval Dataset...")
    # Load dataset
    test_dataset = CustomDataset("test", args.test_path, processor, tokenizer, args.max_seq_len)

    print("Construct Dataloader...")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print("Building SPAN-ASTE model...")
    # get dimension of target and relation
    span_nclass, table_nclass = len(SpanTagging), len(TableTagging)
    # build span-aste model
    model = Ranker(
        args.bert_model,
        span_nclass,
        table_nclass,
        device=device
    )

    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location=torch.device(device)))
    model.to(device)

    metric = SpanEvaluator()

    precision, recall, f1 = evaluate(model, metric, test_dataloader, device)
    print("-----------------------------")
    print("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
          (precision, recall, f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", type=str, default=None, help="The name of bert.")
    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization.")
    args = parser.parse_args()

    do_eval()
