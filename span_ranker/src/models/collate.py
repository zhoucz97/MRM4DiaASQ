#!/usr/bin/env python
# -*- coding: UTF-8 -*-


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

    guid, tok2id, id2tok, input_ids, spans, span_labels, span_type_labels, span_postag_labels = zip(*data)

    return guid, tok2id, id2tok, input_ids, spans, span_labels, span_type_labels, span_postag_labels
