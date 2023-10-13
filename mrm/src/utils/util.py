#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random
import numpy as np
import torch
from functools import partial
from .tager import SentimentTagging
from sklearn.metrics import f1_score, recall_score, precision_score



def cal_pair_score(pred_list, true_list):
    pred_set = set(pred_list)
    true_set = set(true_list)
    correct_num = len(true_set & pred_set)
    precision = correct_num / (len(pred_set) + 1e-8)
    recall = correct_num / (len(true_set) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    precision *= 100
    recall *= 100
    f1 *= 100
    return [precision, recall, f1]


def cal_span_score(pred_list, true_list):
    pass
    

def trans_list(quads, id2tok, conv_len):
    quads = quads[0]
    t_list, a_list, o_list = [0] * conv_len, [0] * conv_len, [0] * conv_len
    for q in quads:
        for i in range(id2tok[q[0]], id2tok[q[1]]+1):
            t_list[i] = 1
        for i in range(id2tok[q[2]], id2tok[q[3]]+1):
            a_list[i] = 1
        for i in range(id2tok[q[4]], id2tok[q[5]]+1):
            o_list[i] = 1
    return t_list, a_list, o_list
        

def trans_str_for_span(spans, guid):
    spans = spans[0]
    span_set = set()
    for span in spans:
        s1, s2 = span
        s = str(guid) + '-' + str([s1, s2])
        span_set.add(s)
    return span_set
    

def trans_str(res, guid):
    t_list, a_list, o_list, ta_list, to_list, ao_list, tao_list, taos_list = set(), set(), set(), set(), set(), set(), set(), set()
    for t in res['targets']:
        string = str(guid) + '-' + str([t[0], t[1]])
        t_list.add(string)
    for t in res['aspects']:
        string = str(guid) + '-' + str([t[0], t[1]])
        a_list.add(string)
    for t in res['opinions']:
        string = str(guid) + '-' + str([t[0], t[1]])
        o_list.add(string)
    for t in res['ta_pairs']:
        string = str(guid) + '-' + str([t[0], t[1], t[2], t[3]])
        ta_list.add(string)
    for t in res['to_pairs']:
        string = str(guid) + '-' + str([t[0], t[1], t[2], t[3]])
        to_list.add(string)
    for t in res['ao_pairs']:
        string = str(guid) + '-' + str([t[0], t[1], t[2], t[3]])
        ao_list.add(string)
    for t in res['triplets']:
        string = str(guid) + '-' + str([t[0], t[1], t[2], t[3], t[4], t[5]])
        tao_list.add(string)
    for t in res['quads']:
        string = str(guid) + '-' + str([t[0], t[1], t[2], t[3], t[4], t[5], t[6]])
        taos_list.add(string)
        
    return [t_list, a_list, o_list, ta_list, to_list, ao_list, tao_list, taos_list]
        

def decode_truth(quads, entities):
    quads = quads[0]
    entities = entities[0]
    t_list, a_list, o_list, ta_list, to_list, ao_list, tao_list, taos_list = set(), set(), set(), set(), set(), set(), set(), set()
    for t in entities['targets']:
        if -1 not in t:
            t_list.add(tuple(t))
    for a in entities['aspects']:
        if -1 not in a:
            a_list.add(tuple(a))
    for o in entities['opinions']:
        if -1 not in o:
            o_list.add(tuple(o))
        
    for q in quads:
        t1, t2, a1, a2, o1, o2, s = q
        t = [t1, t2]
        a = [a1, a2]
        o = [o1, o2]
        ta = [t1, t2, a1, a2]
        to = [t1, t2, o1, o2]
        ao = [a1, a2, o1, o2]
        tao = [t1, t2, a1, a2, o1, o2]
        taos = [t1, t2, a1, a2, o1, o2, s]
        if -1 not in t:
            t_list.add(tuple(t))
        if -1 not in a:
            a_list.add(tuple(a))
        if -1 not in o:
            o_list.add(tuple(o))
        if -1 not in ta:
            ta_list.add(tuple(ta))
        if -1 not in to:
            to_list.add(tuple(to))
        if -1 not in ao:
            ao_list.add(tuple(ao))
        if -1 not in tao:
            tao_list.add(tuple(tao))
        if -1 not in taos:
            taos_list.add(tuple(taos))
    decoding_res = {
        'targets': list(t_list),
        'aspects': list(a_list),
        'opinions': list(o_list),
        'ta_pairs': list(ta_list),
        'to_pairs': list(to_list),
        'ao_pairs': list(ao_list),
        'triplets': list(tao_list),
        'quads': list(taos_list)
    }
    return decoding_res

def id2word(id_list, sentence_idx_list, id2tok):
    res = []
    for i, ids in enumerate(id_list):
        tok_idx = id2tok[ids]
        word_idx = tok_idx - (sentence_idx_list[ids] * 2 + 1)

        if i % 2 != 0:
            word_idx += 1
        res.append(word_idx)
    return res

def decode_prediction(pred_labels, pair_indices, TableLabel, sentence_idx_list, id2tok):
    # ps_labels = ps_labels[0]
    pred_labels = pred_labels[0]  # del batch_size because bs = 1
    pair_indices = pair_indices.cpu().tolist()[0]
    sentence_idx_list = sentence_idx_list.cpu().tolist()[0]
    id2tok = id2tok[0]
    t_list, a_list, o_list = [], [], []
    ta_list, ao_list = [], []
    to_list, to_type_list = [], []
    tao_list, taos_list = [], []
    
    i2w = partial(id2word, sentence_idx_list=sentence_idx_list, id2tok=id2tok)
    
    # decoding target, aspect, opinion
    for i in range(len(pred_labels)):
        if pred_labels[i][i] == TableLabel.TARGET.value:
            t_list.append(pair_indices[i][i][:2])
        elif pred_labels[i][i] == TableLabel.ASPECT.value:
            a_list.append(pair_indices[i][i][:2])
        elif pred_labels[i][i] == TableLabel.OPINION.value:
            o_list.append(pair_indices[i][i][:2])
    # decoding t-a, a-o- t-o pairs
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
            elif pl == TableLabel.AO.value:
                if pi[:2] in a_list and pi[2:] in o_list:
                    ao_list.append(pi)
                elif pi[:2] in o_list and pi[2:] in a_list:
                    ao_list.append(pi[2:] + pi[:2])
            elif pl in [TableLabel.TO_POS.value, TableLabel.TO_NEG.value, TableLabel.TO_NEU.value]:
                if pi[:2] in t_list and pi[2:] in o_list:
                    to_list.append(pi)
                    to_type_list.append(pl)
                elif pi[:2] in o_list and pi[2:] in t_list:
                    to_list.append(pi[2:] + pi[:2])
                    to_type_list.append(pl)
    
    # decoding triplets and quads
    for i in range(len(ta_list)):
        for j in range(len(to_list)):
            ta = ta_list[i]
            to = to_list[j]
            if ta[:2] != to[:2]:
                continue
            tao = ta + to[2:]
            if tao[2:] not in ao_list:
                continue
            # TableLabel.AO_POS.value, TableLabel.AO_NEG.value, TableLabel.AO_NEU
            if to_type_list[j] == TableLabel.TO_POS.value:
                sentiment = 0
            elif to_type_list[j] == TableLabel.TO_NEG.value:
                sentiment = 1
            elif to_type_list[j] == TableLabel.TO_NEU.value:
                sentiment = 2
            else:
                continue
            if tao not in tao_list:
                tao_list.append(tao)
            taos = tao + [sentiment]
            if taos not in taos_list:
                taos_list.append(taos)        
    
    t_list = [i2w(lst) for lst in t_list]
    a_list = [i2w(lst) for lst in a_list]
    o_list = [i2w(lst) for lst in o_list]
    ta_list = [i2w(lst) for lst in ta_list]
    ao_list = [i2w(lst) for lst in ao_list]
    to_list = [i2w(lst) for lst in to_list]
    assert len(tao_list) == len(taos_list)
    tao_list = [i2w(lst) for lst in tao_list]
    
    taos_list = [tao_list[i] + [SentimentTagging._get_sent(taos_list[i][6])] for i in range(len(tao_list))]
    decoding_res = {
        'targets': t_list,
        'aspects': a_list,
        'opinions': o_list,
        'ta': ta_list,
        'to': to_list,
        'ao': ao_list,
        'triplets': taos_list
    }
    return decoding_res
                
    


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
def set_labels(span_pair_indices, quads, TableLabel):
    """construct the table labels 

    Args:
        span_pair_indices (torch.tensor): shape: bs, k, k, 4
        quads (tuple): shape: bs, quad_num

    Returns:
        table_labels: shape: bs, k, k
    """
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
                assert table_labels[i][j] == TableLabel.INVALID.value
                table_labels[i][j] = table_labels[j][i]
    for i, row_labels in enumerate(table_labels):
        for j in range(i):
            table_labels[i][j] = -1
    
    return table_labels

def set_labels2(span_pair_indices, quads, target_spans, aspect_spans, opinion_spans, SpanTypeLabel, TableLabel):
    """construct the table labels 
    Args:
        span_pair_indices (torch.tensor): shape: k, k, 4
        quads (list[tuple]): shape: quad_num, 7

    Returns:
        table_labels: shape: k, k
    """
    span_pair_indices = span_pair_indices.tolist()
    
    targets, aspects, opinions, tas, aos, tos = [], [], [], [], [], []
    sents = []
    
    for t in target_spans:
        if list(t) not in targets:
            targets.append(list(t[:2] * 2))
    for a in aspect_spans:
        if list(a) not in aspects:
            aspects.append(list(a[:2] * 2))
    for o in opinion_spans:
        if list(o) not in opinions:
            opinions.append(list(o[:2] * 2))
    
    for quad in quads:
        target = list(quad[:2] * 2)
        aspect = list(quad[2:4] * 2)
        opinion = list(quad[4:6] * 2)
        ta = list(quad[:4])  # 0-4
        to = list(quad[:2] + quad[4:6])  # 0-2, 4-6
        ao = list(quad[2:6])  # 2-6
        sentiment = quad[6]
        if -1 not in target and target not in targets:
            targets.append(target)
        if -1 not in aspect and aspect not in aspects:
            aspects.append(aspect)
        if -1 not in opinion and opinion not in opinions:
            opinions.append(opinion)
        if -1 not in ta and ta not in tas:
            tas.append(ta)
        if -1 not in ao and ao not in aos:
            aos.append(ao)
        if -1 not in to and to not in tos:
            tos.append(to)
            sents.append(sentiment)
                
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
            elif pair_indice in aos:
                row_labels.append(TableLabel.AO.value)
            elif pair_indice in tos:
                idx = tos.index(pair_indice)
                if sents[idx] == 0:  # pos
                    row_labels.append(TableLabel.TO_POS.value)
                elif sents[idx] == 1:  # neg
                    row_labels.append(TableLabel.TO_NEG.value)
                elif sents[idx] == 2:  # other
                    row_labels.append(TableLabel.TO_NEU.value)
                else:
                    assert 1 == 2
            else:
                row_labels.append(TableLabel.INVALID.value)
        table_labels.append(row_labels)   
    for i, row_labels in enumerate(table_labels):
        for j in range(i+1, len(table_labels)):
            if table_labels[j][i] != TableLabel.INVALID.value:
                assert table_labels[i][j] == TableLabel.INVALID.value
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
    
    
            