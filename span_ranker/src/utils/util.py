#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import random
import numpy as np
import torch


def cal_pair_score(pred_list, true_list):
    pred_set = set(pred_list)
    true_set = set(true_list)
    correct_num = len(true_set & pred_set)
    precision = correct_num / (len(pred_set) + 1e-5)
    recall = correct_num / (len(true_set) + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    precision *= 100
    recall *= 100
    f1 *= 100
    return list([precision, recall, f1])

def cal_span_score(pred_list, true_list):
    pass
    
    
    

def trans_str(quads, guid):
    quads = quads[0]
    res = []
    for q in quads:
        string = str(guid) + '-' + str(q)
        res.append(string)
    return res
        

def decode_quads(pred_labels, pair_indices, TableLabel):
    pred_labels = pred_labels[0]  # del batch_size because bs = 1
    pair_indices = pair_indices.cpu().tolist()[0]
    t_list, a_list, o_list = [], [], []
    ta_list, to_list = [], []
    ao_list, ao_type_list = [], []
    # find target, aspect, opinion
    for i in range(len(pred_labels)):
        if pred_labels[i][i] == TableLabel.TARGET.value:
            t_list.append(pair_indices[i][i][:2])
        elif pred_labels[i][i] == TableLabel.ASPECT.value:
            a_list.append(pair_indices[i][i][:2])
        elif pred_labels[i][i] == TableLabel.OPINION.value:
            o_list.append(pair_indices[i][i][:2])
    for i in range(len(pred_labels)):
        for j in range(i+1, len(pred_labels)):
            pl = pred_labels[i][j]
            pi = pair_indices[i][j]
            if pl == TableLabel.INVALID.value:
                continue
            elif pl == TableLabel.TA.value:
                if pi[:2] in t_list and pi[2:] in a_list:
                    ta_list.append(pi)
                elif pi[:2] in a_list and pi[2:] in t_list:
                    ta_list.append(pi[2:] + pi[:2])
            elif pl == TableLabel.TO.value:
                if pi[:2] in t_list and pi[2:] in o_list:
                    to_list.append(pi)
                elif pi[:2] in o_list and pi[2:] in t_list:
                    to_list.append(pi[2:] + pi[:2])
            elif pl in [TableLabel.AO_POS.value, TableLabel.AO_NEG.value, TableLabel.AO_NEU]:
                if pi[:2] in a_list and pi[2:] in o_list:
                    ao_list.append(pi)
                    ao_type_list.append(pl)
                elif pi[:2] in o_list and pi[2:] in a_list:
                    ao_list.append(pi[2:] + pi[:2])
                    ao_type_list.append(pl)
                
    pred_quads = []      
    for i in range(len(ta_list)):
        for j in range(len(ao_list)):
            ta = ta_list[i]
            ao = ao_list[j]
            if ta[2:] != ao[:2]:
                continue
            tao = ta + ao[2:]
            TableLabel.AO_POS.value, TableLabel.AO_NEG.value, TableLabel.AO_NEU
            if ao_type_list[j] == TableLabel.AO_POS.value:
                sentiment = 0
            elif ao_type_list[j] == TableLabel.AO_NEG.value:
                sentiment = 1
            else:
                sentiment = 2
            taos = tao + [sentiment]
            to = tao[:2] + tao[4:]
            if to not in to_list:
                continue
            if taos not in pred_quads:
                pred_quads.append(taos)
    return [pred_quads]
                
    


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
def set_labels(span_pair_indices, quads, TableLabel):

    span_pair_indices = span_pair_indices.cpu().tolist()
    span_pair_indices = span_pair_indices[0]  # del batch because batch_size = 1
    quads = quads[0]
    
    targets, aspects, opinions, tas, tos, aos = [], [], [], [], [], []
    ao_types = []
    for quad in quads:
        target = list(quad[:2] * 2)
        aspect = list(quad[2:4] * 2)
        opinion = list(quad[4:6] * 2)
        ta = list(quad[:4])
        to = list(quad[:2] + quad[4:6])
        ao = list(quad[2:6])
        ao_type = quad[6]
        if target not in targets:
            targets.append(target)
        if aspect not in aspects:
            aspects.append(aspect)
        if opinion not in opinions:
            opinions.append(opinion)
        if ta not in tas:
            tas.append(ta)
        if to not in tos:
            tos.append(to)
        if ao not in aos:
            aos.append(ao)
            ao_types.append(ao_type)
    
    table_labels = []
    for rows in span_pair_indices:
        row_labels = []
        for span_pair_indice in rows: # [span1.s, span1.e, span2.s, span2.t]
            pair_indice = span_pair_indice
            if pair_indice in targets:
                row_labels.append(TableLabel.TARGET.value)
            elif pair_indice in aspects:
                row_labels.append(TableLabel.ASPECT.value)
            elif pair_indice in opinions:
                row_labels.append(TableLabel.OPINION.value)
            elif pair_indice in tas:
                row_labels.append(TableLabel.TA.value)
            elif pair_indice in tos:
                row_labels.append(TableLabel.TO.value)
            elif pair_indice in aos:
                idx = aos.index(pair_indice)
                if ao_types[idx] == 0:  # pos
                    row_labels.append(TableLabel.AO_POS.value)
                elif ao_types[idx] == 1:  # pos
                    row_labels.append(TableLabel.AO_NEG.value)
                elif ao_types[idx] == 2:  # pos
                    row_labels.append(TableLabel.AO_NEU.value)
                else:
                    assert 1 == 2
            else:
                row_labels.append(TableLabel.INVALID.value)
        table_labels.append(row_labels)   
    for i, row_labels in enumerate(table_labels):
        for j in range(i+1, len(table_labels)):
            if table_labels[j][i] != TableLabel.INVALID.value:
                table_labels[i][j] = table_labels[j][i]
    for i, row_labels in enumerate(table_labels):
        for j in range(i):
            table_labels[i][j] = -1
    
    return table_labels




from enum import IntEnum
class TableLabel(IntEnum):
    INVALID = 0
    TARGET = 1
    ASPECT = 2
    OPINION = 3
    TA = 4
    TO = 5
    AO_POS = 6
    AO_NEG = 7
    AO_NEU = 8
    
if __name__ == '__main__':
    pairs = []
    for s1 in [[1,2], [5,6], [3,4], [7,8], [1,5]]:
        p = []
        for s2 in [[1,2], [5,6], [3,4], [7,8], [1,5]]:
            p.append(s1 + s2)
        pairs.append(p)
    print(pairs)
    quads = [[[1,2,3,4,7,8,2]]]
    set_labels(torch.tensor([pairs]), quads, TableLabel)
    
    
            