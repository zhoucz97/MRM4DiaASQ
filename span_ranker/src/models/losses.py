#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn


def log_likelihood(preds, trues, device):
    bs, seq_len, n_class = preds.size()
    
    preds = preds.view(bs * seq_len, n_class)
    trues = torch.LongTensor(trues).to(device).view(bs * seq_len)
    
    # weight=torch.FloatTensor([1, 2]).to(device)
    criterion = nn.NLLLoss(ignore_index=-1)
    return criterion(torch.log(preds), trues)

