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
from models.model import Model
from utils.dataset import CustomDataset
from utils.processor import QuadDataProcessor
from utils.tager import SpanTagging, TableTagging
from evals import Template
from utils.util import decode_prediction, decode_truth, trans_str, cal_pair_score, cal_span_score, trans_str_for_span
from sklearn.metrics import f1_score, precision_score, recall_score



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate(gold_file_path, model, data_loader, device):
    model.eval()
    topk_sum, all_sum = 0, 0
    pred_res_list = [[], [], [], [], [], [], [], []]
    true_res_list = [[], [], [], [], [], [], [], []]
    pred_span_list = [[], [], []]
    true_span_list = [[], [], []]
    preds = []
    template = Template(gold_file_path)
    with torch.no_grad():
        for batch_ix, batch in enumerate(data_loader):
            guid, tok2id, id2tok, input_ids, attn_masks, span_indices, span_labels, span_type_labels, span_postag_labels, quads, \
                ranker_result, topk_span_indices_idx, table_labels, _, mask_rep, mask_spe, entities, sentence_idx_list = batch
            span_pair_probs, span_pair_indices = model(input_ids, attn_masks, span_indices, 
                                                                   span_postag_labels, span_labels, topk_span_indices_idx, 
                                                                   mask_rep, mask_spe)
            guid = guid[0]

            span_pair_labels = torch.argmax(span_pair_probs, dim=-1).cpu().tolist() # shape: bs, pair_num, pair_num, nclass
            # span_labels = torch.argmax(span_probs, dim=-1).cpu().tolist() # shape: bs, pair_num, pair_num, nclass
            # obtain the index of target/aspect/opinion spans, ta/to/ao-pairs, tao-triplets and taos quads.
            pred_res = decode_prediction(span_pair_labels, span_pair_indices, TableTagging, sentence_idx_list, id2tok)
            # true_res = decode_truth(quads, entities)
            pred_res['doc_id'] = guid
            preds.append(pred_res)
            

            
    
    micro_score, iden_score, res = template.forward(preds)

    return micro_score, iden_score, res


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
    model = Model(
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
