#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn


def loss_func(preds, trues, mask, device):
    bs, sp_num, _, n_class = preds.size()
    
    preds = preds.view(bs * sp_num * sp_num, n_class)
    trues = torch.LongTensor(trues).to(device).view(bs * sp_num * sp_num)
    mask = mask.to(device).view(bs * sp_num * sp_num)
    trues = trues.where(mask > 0, -torch.ones(trues.size()).to(device).long())

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    return criterion(preds, trues)


def log_likelihood(preds, trues, device, nclass, mask=None):
    preds = preds.view(-1, nclass)
    trues = torch.LongTensor(trues).to(device).view(-1)
    if mask != None:
        mask = mask.view(-1)
        preds = preds[mask, :]
        trues = trues[mask]
    
    weight = torch.FloatTensor([1] + [5] * (nclass-1)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

    return criterion(preds, trues)
