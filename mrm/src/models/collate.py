#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch.nn.utils.rnn import pad_sequence


def gold_labels(span_indices, spans, span_labels):

    # gold span labels
    gold_indices, gold_labels = [], []
    for batch_idx, span_b in enumerate(spans):
        for span in span_b:
            if span not in span_indices[batch_idx]:
                # print(span)
                pass
                
    for batch_idx, indices in enumerate(span_indices):
        gold_ind, gold_lab = [], []
        for indice in indices:
            # if indice == (-1, -1):
            #     gold_lab.append(-1)
            if indice in spans[batch_idx]:
                ix = spans[batch_idx].index(indice)
                gold_lab.append(span_labels[batch_idx][ix])
            else:
                gold_lab.append(0)
            gold_ind.append(indice)
        gold_indices.append(gold_ind)
        gold_labels.append(gold_lab)

    return gold_indices, gold_labels


def collate_fn(data):

    guid, tok2id, id2tok, input_ids, spans, span_labels, span_type_labels, span_postag_labels, quads, ranker_result, \
        topk_span_indices_idx, table_labels, span_pair_indices, mask_rep, mask_spe, entities, sentence_idx_list = zip(*data)
    
    max_len = max([len(x) for x in input_ids])
    max_span_num = max([len(span) for span in spans])
    
    spans = [span + [(0, 0)] * (max_span_num-len(span)) for span in spans]
    spans = torch.tensor(spans)
    
    topk_span_indices_idx = torch.tensor(topk_span_indices_idx)  # bs, top_k
    
    # func = lambda fx: list(map(lambda x: torch.LongTensor(x), fx))
    input_ids = [torch.LongTensor(x) for x in input_ids]
    mask_rep = [torch.LongTensor(x) for x in mask_rep]
    mask_spe = [torch.LongTensor(x) for x in mask_spe]
    sentence_idx_list = [torch.LongTensor(x) for x in sentence_idx_list] 
    
    # padding
    sentence_idx_list = torch.nn.utils.rnn.pad_sequence(sentence_idx_list, batch_first=True, padding_value=-1)
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)
    attn_masks = input_ids != 1
    mask_rep = list(torch.nn.functional.pad(x, (0, max_len-x.size(0), 0, max_len-x.size(0)), 'constant', 0) for x in mask_rep)
    mask_spe = list(torch.nn.functional.pad(x, (0, max_len-x.size(0), 0, max_len-x.size(0)), 'constant', 0) for x in mask_spe)
    mask_rep = torch.stack(mask_rep)
    mask_spe = torch.stack(mask_spe)
    
    max_labels_len = max([len(table_label[0]) for table_label in table_labels])
    table_labels = [torch.LongTensor(x) for x in table_labels]
    table_labels = list(torch.nn.functional.pad(x, (0, max_labels_len-x.size(0), 0, max_labels_len-x.size(0)), 'constant', -1) for x in table_labels)
    table_labels = torch.stack(table_labels)
    
    return guid, tok2id, id2tok, input_ids, attn_masks, spans, span_labels, span_type_labels, span_postag_labels, quads, \
        ranker_result, topk_span_indices_idx, table_labels, span_pair_indices, mask_rep, mask_spe, entities, sentence_idx_list
