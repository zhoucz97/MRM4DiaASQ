#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init

from transformers import AutoModel

from utils.tager import SpanTypeTagging, PosTaging, ZHPosTaging


class SpanRepresentation(nn.Module):
    def __init__(self, span_width_embedding_dim, span_postag_embedding_dim):
        super(SpanRepresentation, self).__init__()
        # self.span_maximum_length = span_maximum_length
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 31, 32, 63, 64]
        self.span_width_embedding = nn.Embedding(len(self.bucket_bins), span_width_embedding_dim)
        self.span_pos_embedding = nn.Embedding(ZHPosTaging.pos_num + 1, span_postag_embedding_dim)

    def bucket_embedding(self, width, device):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.span_width_embedding(torch.LongTensor([em]).to(device))

    def forward(self, x: Tensor, spans_indices, span_pt_labels):
        batch_size, sequence_length, _ = x.size()
        device = x.device
    
        x_spans = []
        for s in spans_indices[0]:
            pt0 = torch.LongTensor([span_pt_labels[0][int(s[0])]]).to(device)
            pt1 = torch.LongTensor([span_pt_labels[0][int(s[1])]]).to(device)
            postag_emb0 = self.span_pos_embedding(pt0).repeat((batch_size, 1))
            postag_emb1 = self.span_pos_embedding(pt1).repeat((batch_size, 1))
            width_emb = self.bucket_embedding(abs(s[1] - s[0] + 1), device).repeat((batch_size, 1))
            mean_vec = torch.mean(x[:, s[0]: s[1]+1, :], dim=1)
            x_spans.append(torch.cat((mean_vec, x[:, s[0], :], postag_emb0, x[:, s[1], :], postag_emb1, width_emb), dim=-1))

        x_spans = torch.stack(x_spans, dim=1)
        return x_spans


class PrunedTargetOpinion:

    def __init__(self):
        pass

    def __call__(self, span_probs, nz):
        target_indices = torch.topk(span_probs[:, :, SpanTypeTagging.TARGET.value], nz, dim=-1).indices
        aspect_indices = torch.topk(span_probs[:, :, SpanTypeTagging.ASPECT.value], nz, dim=-1).indices
        opinion_indices = torch.topk(span_probs[:, :, SpanTypeTagging.OPINION.value], nz, dim=-1).indices
        return target_indices, aspect_indices, opinion_indices


class Ranker(nn.Module):
    def __init__(
            self,
            pretrain_model,
            span_nclass: "int",
            span_type_nclass: "int",
            table_nclass: "int",
            ffnn_hidden_dim: "int" = 768,
            span_width_embedding_dim: "int" = 256,
            span_postag_embedding_dim: "int" = 256,
            device="cpu"
    ) -> None:
        super(Ranker, self).__init__()
        self.device = device
        self.bert = AutoModel.from_pretrained(pretrain_model)
        encoding_dim = self.bert.config.hidden_size
        span_dim = encoding_dim * 3 + span_width_embedding_dim + span_postag_embedding_dim * 2

        self.span_representation = SpanRepresentation(span_width_embedding_dim, span_postag_embedding_dim)
        self.span_ffnn = torch.nn.Sequential(
            nn.Linear(span_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(ffnn_hidden_dim, span_nclass, bias=True),
            nn.Softmax(-1)
        )
        self.span_type_ffnn = torch.nn.Sequential(
            nn.Linear(span_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(ffnn_hidden_dim, span_type_nclass, bias=True),
            nn.Softmax(-1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.span_ffnn.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)
        for name, param in self.span_type_ffnn.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)

    def forward(self, input_ids, span_indices, span_postag_labels):
        span_indices = torch.tensor(span_indices).to(self.device)
        bs, span_num, _ = span_indices.size()
        
        bert_states = self.bert(input_ids).last_hidden_state # shape: 1, token_num, 768
        span_states = self.span_representation(bert_states, span_indices, span_postag_labels)  # shape: bs, span_num, dim
        span_probs = self.span_ffnn(span_states)
        span_type_probs = self.span_type_ffnn(span_states)
        assert span_probs.size(1) == span_num
        assert span_type_probs.size(1) == span_num
        
        return span_probs, span_type_probs
