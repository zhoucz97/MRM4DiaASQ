#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import LSTM, init
import itertools
import math

from transformers import BertModel, AutoModel

from utils.tager import SpanTypeTagging, PosTaging, ZHPosTaging


class SpanRepresentation(nn.Module):
    def __init__(self, span_width_embedding_dim, span_postag_embedding_dim, dataset_lan='en'):
        super(SpanRepresentation, self).__init__()
        if dataset_lan == 'zh':
            tager = ZHPosTaging
        else:
            tager = PosTaging
        # self.span_maximum_length = span_maximum_length
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        self.span_width_embedding = nn.Embedding(len(self.bucket_bins), span_width_embedding_dim)
        self.span_pos_embedding = nn.Embedding(tager.pos_num + 1, span_postag_embedding_dim)

    # def bucket_embedding(self, width, device):
    #     em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
    #     return self.span_width_embedding(torch.LongTensor([em]).to(device))

    def forward(self, x: Tensor, span_indices):
        bs, seq_len, x_dim = x.size()
        device = x.device
        begin_indices = span_indices[:, :, 0]  # shape: bs, span_num
        end_indices = span_indices[:, :, 1]
        begin_states = torch.gather(input=x, dim=1, index=begin_indices.unsqueeze(-1).repeat(1, 1, x_dim))  # shape: bs, span_num, x_dim
        end_states = torch.gather(input=x, dim=1, index=end_indices.unsqueeze(-1).repeat(1, 1, x_dim))
        
        widths = end_indices - begin_indices  # shape: bs, span_num
        widths = torch.where(widths > 10, 11, widths)  # 将大于10的宽度统计为11
        width_emb = self.span_width_embedding(widths)  # shape: bs, span_num, width_emb_dim
        
        x_spans = torch.cat([begin_states, end_states], dim=-1)
        

        
        
        return x_spans


class PrunedTargetOpinion:


    def __init__(self):
        pass

    def __call__(self, span_probs, nz):
        target_indices = torch.topk(span_probs[:, :, SpanTypeTagging.TARGET.value], nz, dim=-1).indices
        aspect_indices = torch.topk(span_probs[:, :, SpanTypeTagging.ASPECT.value], nz, dim=-1).indices
        opinion_indices = torch.topk(span_probs[:, :, SpanTypeTagging.OPINION.value], nz, dim=-1).indices
        return target_indices, aspect_indices, opinion_indices


class PairRepresentation(nn.Module):
    def __init__(self, distance_embeddings_dim):
        super(PairRepresentation, self).__init__()
        # self.bucket_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 31, 32, 63, 64]
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [10] * 5 + [15] + [16] * 15 + [31] + [32] * 31 + [63] + [64] * 500
        self.distance_embeddings = nn.Embedding(len(self.bucket_bins), distance_embeddings_dim)

    def forward(self, span_pair_indices, topk_span_states, k):
        batch_size, span_num, feat_dim = topk_span_states.size()
        device = topk_span_states.device
        
        # obtain span_pair_indices
        # topk_span_indices = topk_span_indices.unsqueeze(2).expand([-1, -1, k, -1])
        # topk_span_indices_T = topk_span_indices.transpose(1, 2)
        # span_pair_indices = torch.cat([topk_span_indices, topk_span_indices_T], dim=-1)
        # obtain span_pair_states
        topk_span_states = topk_span_states.unsqueeze(2).expand([-1, -1, k, -1])  # bs, topk, topk, dim
        topk_span_states_T = topk_span_states.transpose(1, 2)
        span_pair_states = torch.cat([topk_span_states, topk_span_states_T, topk_span_states * topk_span_states_T], dim=-1)  # bs, topk, topk, dim * 2
        
        min_dis = torch.min(
            torch.stack(
                [torch.abs(span_pair_indices[:, :, :, 0] - span_pair_indices[:, :, :, 3]), 
                 torch.abs(span_pair_indices[:, :, :, 1] - span_pair_indices[:, :, :, 2])],
                dim=-1), 
            dim=-1
            ).values  # shape: bs, pair_num, pair_num
        min_dis = min_dis.reshape(batch_size, span_num**2) # shape: bs, pair_num*pair_num
        # shape: bs, span_num**2, len(self.bucket_bins)
        bucket_bins = torch.tensor(self.bucket_bins).unsqueeze(0).unsqueeze(0).repeat(batch_size, span_num**2, 1).to(device)
        min_dis2 = torch.gather(bucket_bins, -1, min_dis.unsqueeze(-1)).squeeze(2) # shape: bs, span_num**2
        dis_embeddings = self.distance_embeddings(min_dis2) # bs, span_num**2, embed_dim
        
        # span_pair_indices = span_pair_indices.reshape(batch_size, span_num**2, 4)
        span_pair_states = span_pair_states.reshape(batch_size, span_num**2, -1)
        # pair_states = torch.cat([span_pair_states, dis_embeddings], dim=-1) # bs, span_num**2, dim+embed_dim
        pair_states = span_pair_states
        pair_states = pair_states.reshape(batch_size, span_num, span_num, -1)
        return pair_states


class Model(nn.Module):
    def __init__(
            self,
            pretrain_model,
            span_nclass: "int",
            span_type_nclass: "int",
            table_nclass: "int",
            ffnn_hidden_dim: "int" = 768,
            span_width_embedding_dim: "int" = 128,
            span_postag_embedding_dim: "int" = 128,
            device="cpu",
            args=None
    ) -> None:
        super(Model, self).__init__()
        self.dataset_lan = args.lan
        self.device = device
        self.bert = AutoModel.from_pretrained(pretrain_model)
        encoding_dim = self.bert.config.hidden_size
        # span_dim = encoding_dim * 2 + span_width_embedding_dim + span_postag_embedding_dim * 2
        # span_dim = encoding_dim * 2 + span_width_embedding_dim
        table_dim = encoding_dim * 3
        
        att_head_size = int(encoding_dim / 4)
        self.self_attn_reply = MultiHeadAttention(4, encoding_dim, att_head_size, att_head_size)
        self.self_attn_speaker = MultiHeadAttention(4, encoding_dim, att_head_size, att_head_size)
        self.span_representation = SpanRepresentation(span_width_embedding_dim, span_postag_embedding_dim, self.dataset_lan)
        self.span_mapping = nn.Linear(encoding_dim * 2, encoding_dim)
        self.pair_representation = PairRepresentation(distance_embeddings_dim=span_width_embedding_dim)
        self.table_ffnn = torch.nn.Sequential(
            nn.Linear(table_dim, ffnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(ffnn_hidden_dim, table_nclass),
        )
        # self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.table_ffnn.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)
        for name, param in self.span_mapping.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)

    def forward(self, input_ids, attn_masks, span_indices, span_postag_labels, span_labels, topk_span_indices_idx, mask_rep, mask_spe):
        input_ids = torch.LongTensor(input_ids).to(self.device)  # shape: bs, ids_num
        attn_masks = torch.BoolTensor(attn_masks).to(self.device)
        mask_rep = torch.LongTensor(mask_rep).to(self.device)
        mask_spe = torch.LongTensor(mask_spe).to(self.device)
        span_indices = torch.LongTensor(span_indices).to(self.device)  # shape: bs, span_num, 2
        topk_span_indices_idx = torch.LongTensor(topk_span_indices_idx).to(self.device) # shape: bs, k

        k = topk_span_indices_idx.size(1)
        
        # obtain span_pair_indices
        topk_span_indices = span_indices.gather(dim=1, index=topk_span_indices_idx.unsqueeze(-1).repeat(1, 1, span_indices.size(-1)))
        topk_span_indices = topk_span_indices.unsqueeze(2).expand([-1, -1, k, -1])
        span_pair_indices = torch.cat([topk_span_indices, topk_span_indices.transpose(1, 2)], dim=-1) # shape: bs, k, k, 4

        bert_states = self.bert(input_ids=input_ids, attention_mask=attn_masks).last_hidden_state # shape: bs, ids_num, plm.hidden_size
        bert_states_rep = self.self_attn_reply(bert_states, bert_states, bert_states, mask_rep)
        bert_states_spe = self.self_attn_speaker(bert_states, bert_states, bert_states, mask_spe)
        bert_states = torch.max(torch.stack((bert_states_rep, bert_states_spe), 0), 0)[0]

        span_states = self.span_mapping(self.span_representation(bert_states, span_indices))  # shape: bs, span_num, dim
        topk_span_states = span_states.gather(dim=1, index=topk_span_indices_idx.unsqueeze(-1).repeat(1, 1, span_states.size(-1)))  # bs, topk, hidden_size
        
        span_pair_states = self.pair_representation(span_pair_indices, topk_span_states, k)
        span_pair_probs = self.table_ffnn(span_pair_states)
        
        return span_pair_probs, span_pair_indices



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.5)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)  # bs, num, hidden_size
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)  # bs, num_attn_heads, num, attn_heads_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # bs, num_attn_heads, num, num
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # attention_mask = torch.logical_not(attention_mask) * -999999
            # attention_scores = attention_scores + attention_mask
            attention_scores.masked_fill_(attention_mask.bool(), -99999)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = self.LayerNorm(self.dense(self.dropout(context_layer))) + hidden_states
        return outputs


class GraphAttentionLayer(nn.Module):

    def __init__(self, att_head, in_dim, dp_gnn=0.5, leaky_alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = int(in_dim / att_head)
        self.dp_gnn = dp_gnn
        self.att_head = att_head
        assert self.in_dim == self.out_dim*self.att_head
        self.leaky_alpha = leaky_alpha
        
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))
        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.H = nn.Linear(self.in_dim, self.in_dim)
        
        self.init_gnn_param()
        
    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)
        init.xavier_normal_(self.H.weight)

    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim

        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)

        attn_src = torch.matmul(torch.tanh(h), self.w_src)
        attn_dst = torch.matmul(torch.tanh(h), self.w_dst)
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        if adj != None:
            mask = torch.logical_not(adj.unsqueeze(1))
            # mask = 1 - adj.unsqueeze(1)
            attn.data.masked_fill_(mask.bool(), -999)
        attn = F.softmax(attn, dim=-1)

        feat_out = torch.matmul(attn, h) + self.b
        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = torch.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'