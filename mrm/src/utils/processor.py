#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy
import json
import os
from enum import IntEnum
import nltk
import pickle
import torch
from tqdm import tqdm
import jieba.posseg as psg

from utils.tager import SpanTagging, SpanTypeTagging, TableTagging, PosTaging, ZHPosTaging, SentimentTagging
from utils.util import set_labels2


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, dialog, input_ids, sentences_num, sentences_len_list, sentence_idx_list,
                 spans, span_labels, span_type_labels, quads, tok2id, id2tok, span_postag_labels,
                 ranker_result, topk_span_indices_idx, table_labels, span_pair_indices,
                 mask_rep, mask_spe, entities):
        """Constructs a InputExample."""
        self.guid = guid
        self.dialog = dialog
        self.input_ids = input_ids
        self.sentences_num = sentences_num
        self.sentences_len_list = sentences_len_list
        self.spans = spans
        self.span_labels = span_labels
        self.span_type_labels = span_type_labels
        self.quads = quads
        self.tok2id = tok2id
        self.id2tok = id2tok
        self.span_postag_labels = span_postag_labels
        self.ranker_result = ranker_result
        self.topk_span_indices_idx = topk_span_indices_idx
        self.table_labels = table_labels
        self.span_pair_indices = span_pair_indices
        self.mask_rep = mask_rep
        self.mask_spe = mask_spe
        self.entities = entities
        self.sentence_idx_list = sentence_idx_list

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len


def get_spans(tags):
    '''for BIO tag'''
    tags = [t for tag in tags for t in tag]
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('\\B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('\\O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class QuadDataProcessor():
    def __init__(self, tokenizer, max_length, args):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.span_max_len = 10
        self.pos_tagger = nltk.pos_tag
        self.args = args

    def get_examples(self, data_dir, data_type):
        """See base class."""
        cache_file_path = os.path.join(self.args.cache_path, str(data_type)+str(self.args.top_k)+'.pkl')
        # examples = self._create_examples(data_dir, data_type)
        if not os.path.exists(cache_file_path):
            print('The {} cache does not exist, preprocess the {} data....'.format(data_type, data_type))
            examples = self._create_examples(data_dir, data_type)
            with open(cache_file_path, 'wb') as cache_file:
                pickle.dump(examples, cache_file)
        else:
            print('The {} cache exists, load {} data from {}'.format(data_type, data_type, cache_file_path))
            with open(cache_file_path, 'rb') as cache_file:
                examples = pickle.load(cache_file)
        return examples
    
    def _add_spec_token_idx(self, span, conv_lens):
        # 返回给每个句子添加了CLS, SEP之后的index
            # spans: [a, b)
            # res: [a, b]
        if list(span) == [-1, -1]:
            return [-1, -1]
        res = [-1, -1]
        sumlen = 0
        for i, conv_len in enumerate(conv_lens):
            sumlen += conv_len
            if span[0] < sumlen:
                res[0] = 2 * i + 1 + span[0]
            if span[1] - 1 < sumlen:
                res[1] = 2 * i + 1 + span[1]
                break
        return res    
    
    
    def _obtain_tags(self, sent_list, l, r):
        if l == -1 and r == -1:
            return [], -1
        tags = []
        idx = 0
        flag = -1
        for i, sent in enumerate(sent_list):
            t = []
            for word in sent:
                if idx == l:
                    t.append(word+'\\B')
                    flag = i
                elif l < idx < r:
                    t.append(word+'\\I')
                    if flag != i:
                        import pdb
                        pdb.set_trace()
                        assert 1 == 2
                else:
                    t.append(word+'\\O')
                idx += 1
            tags.append(t)
            
        return tags, flag
    
    def _read_orig_files(self, data_dir, data_type):
        print(data_dir)
        with open(data_dir) as file:
            datas = json.load(file)

        outs = []
        for d in datas:
            id = d['doc_id']
            cls_token = [self.tokenizer.cls_token]
            end_token = [self.tokenizer.sep_token]
            sent_list = []
            for sen in d['sentences']:
                sent_list.append([' '+w for w in sen.split()])
                sent_list[-1][0] = sent_list[-1][0].strip()
            orig_sent_lens = [len(s) for s in sent_list]
            sent_list = [cls_token+s+end_token for s in sent_list]
            # sentences = ' '.join(d['sentences']).split(' ')
            replies = d['replies']
            speakers = d['speakers']
            triplets = d['triplets']
            # triplets = [t for t in triplets if -1 not in t[:6]] 
            triplets = [t for t in triplets]
            
            
            targets, aspects, opinions = [], [], []
            for i, t in enumerate(d['targets']):
                t1, t2 = t[0:2]
                if t1 == -1: 
                    continue
                nt1, nt2 = self._add_spec_token_idx([t1, t2], orig_sent_lens)
                targets.append([nt1, nt2])
            for i, t in enumerate(d['aspects']):
                t1, t2 = t[0:2]
                if t1 == -1: 
                    continue
                nt1, nt2 = self._add_spec_token_idx([t1, t2], orig_sent_lens)
                aspects.append([nt1, nt2])
            for i, t in enumerate(d['opinions']):
                t1, t2 = t[0:2]
                if t1 == -1: 
                    continue
                nt1, nt2 = self._add_spec_token_idx([t1, t2], orig_sent_lens)
                opinions.append([nt1, nt2])
                
                
            quads = []
            for i, t in enumerate(triplets):
                t1, t2, a1, a2, o1, o2, sentiment = t[0:7]
                nt1, nt2 = self._add_spec_token_idx([t1, t2], orig_sent_lens)
                na1, na2 = self._add_spec_token_idx([a1, a2], orig_sent_lens)
                no1, no2 = self._add_spec_token_idx([o1, o2], orig_sent_lens)
                target_tags, t_sent_idx = self._obtain_tags(sent_list, nt1, nt2)  # t_sent_idx是target所在句子索引
                aspect_tags, a_sent_idx = self._obtain_tags(sent_list, na1, na2)
                opinion_tags, o_sent_idx = self._obtain_tags(sent_list, no1, no2)
                if sentiment == 'pos':
                    sentiment = 'positive'
                elif sentiment == 'neg':
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                dic = {
                    'uid': str(id)+'-'+str(i),
                    'target_tags': [nt1, nt2],
                    'aspect_tags': [na1, na2],
                    'opinion_tags': [no1, no2],
                    'sentiment': sentiment,
                    'locate_sent_idx': [t_sent_idx, a_sent_idx, o_sent_idx]
                }
                quads.append(dic)

            assert len(replies) == len(sent_list) and len(speakers) == len(sent_list)
            d_dic = {
                'id': id,
                'sentence': sent_list,
                'quads': quads,
                'replies': replies,
                'speakers': speakers,
                'entities': {'targets': targets, 'aspects': aspects, 'opinions': opinions}
            }
            outs.append(d_dic)

        return outs
    
    def _create_examples(self, data_dir, data_type):
        """Creates examples for the training and dev sets."""
        cs_num = 0
        sss_num, all_sss_num = 0, 0
        
        with open(os.path.join(self.args.ranker_path, str(self.args.top_k), str(data_type)+'.pkl'), 'rb') as pkl_file:
            ranker_result = pickle.load(pkl_file)
        
        dialogs = self._read_orig_files(data_dir, data_type)

        tar_num, asp_num, opi_num = 0, 0, 0
        maxtar_num, maxasp_num, maxopi_num = 0, 0, 0
        mintar_num, minasp_num, minopi_num = 999, 999, 999
        examples = []
        for dialog in dialogs:
            dialog_id = dialog['id']
            sentences = dialog['sentence']
            if 'zh' in data_dir:
                sentences = [[w.strip() for w in sent] for sent in sentences]
            quads = dialog['quads']
            replies = dialog['replies']
            speakers = dialog['speakers']
            
            targets = dialog['entities']['targets']
            aspects = dialog['entities']['aspects']
            opinions = dialog['entities']['opinions']
            
            dialog_len = len([s for sents in sentences for s in sents])
            
            # set pos tag for each word
            pos_tag_list, pos_label_list = [], []
            if 'en' in data_dir:  # english dataset 
                for i, sent in enumerate(sentences):
                    sent = [w.strip() for w in sent]  # del whitespace
                    pos_tag_list.append(PosTaging.get_sentence_posTag(self.pos_tagger, sent))
                for pts in pos_tag_list:
                    pos_label_list.append([PosTaging.get_label(pt) for pt in pts])
            else:  # chinese dataset
                self.pos_tagger = psg
                for i, sent in enumerate(sentences):
                    sent = [w.strip() for w in sent]  # del whitespace
                    pos_tag_list.append(ZHPosTaging.get_sentence_posTag(self.pos_tagger, sent))
                for pts in pos_tag_list:
                    pos_label_list.append([ZHPosTaging.get_label(pt) for pt in pts])
                    
            assert len(sentences) == len(pos_label_list)
            for i in range(len(sentences)):
                assert len(sentences[i]) == len(pos_label_list[i])                
            
            tok2id = []
            id2tok = []
            token_start = 0
            cls_idx_list = []
            sep_idx_list = []
            pos_label_uq_list = []
            is_suffix_list = []
            bert_tokens = []
            k = 0
            for i, sent in enumerate(sentences):
                cls_idx_list.append(token_start)
                if i != 0:
                    sep_idx_list.append(token_end)
                for j, w in enumerate(sent):
                    tokens = self.tokenizer.encode(w, add_special_tokens=False)
                    pos_label_uq_list.extend([pos_label_list[i][j]] * len(tokens))
                    is_suffix_list.extend([0] + [1] * (len(tokens) - 1))
                    bert_tokens.extend(tokens)
                    token_end = token_start + len(tokens) - 1
                    tok2id.append([token_start, token_end])
                    id2tok.extend([k] * (token_end - token_start + 1))
                    token_start = token_end + 1
                    k += 1
            sep_idx_list.append(token_end)
            bert_len = len(bert_tokens)
            assert len(is_suffix_list) == bert_len
            assert len(pos_label_uq_list) == bert_len
            assert len(tok2id) == dialog_len
            assert len(id2tok) == bert_len
            
            
            sentence_idx_list = []
            for i in range(len(cls_idx_list)):
                sentence_idx_list.extend([i] * (sep_idx_list[i] - cls_idx_list[i] + 1))
            
            # obtain the mask_repies and mask_speakers matrix
            rep_list, spe_list = [], []
            mask_rep, mask_spe = [], []
            for i in range(len(cls_idx_list)):
                # -999 denote the location of cls and sep.
                rep_list.extend([-999] + [replies[i]] * (sep_idx_list[i] - cls_idx_list[i] - 1) + [-999])
                spe_list.extend([-999] + [speakers[i]] * (sep_idx_list[i] - cls_idx_list[i] - 1) + [-999])
            assert len(rep_list) == bert_len and len(spe_list) == bert_len
            for i in range(len(rep_list)):
                temp_rep, temp_spe = [], []
                for j in range(len(rep_list)):
                    if rep_list[i] == -999 or rep_list[j] == -999:
                        temp_rep.append(0)
                    elif rep_list[i] + 1 == rep_list[j]:  # j is a reply of i.
                        temp_rep.append(1)
                    else:
                        temp_rep.append(0)
                    if spe_list[i] == -999 or spe_list[j] == -999:
                        temp_spe.append(0)
                    elif spe_list[i] == spe_list[j]:  # smae speaker
                        temp_spe.append(1)
                    else:
                        temp_spe.append(0)
                assert len(temp_rep) == bert_len and len(temp_spe) == bert_len
                mask_rep.append(temp_rep)
                mask_spe.append(temp_spe)                      
            assert len(mask_rep) == bert_len and len(mask_spe) == bert_len


            candidate_spans = []
            '''

            '''
            for i in range(len(cls_idx_list)):
                begin_idx = cls_idx_list[i] + 1
                end_idx = sep_idx_list[i]
                # if is_suffix_list[begin_idx] or pos_label_uq_list[begin_idx] == PosTaging.pos_num:
                    # continue
                if is_suffix_list[begin_idx]:
                    continue
                for bi in range(begin_idx, end_idx):
                    for end in range(bi, min(bi+self.span_max_len, end_idx)):
                        # if pos_label_uq_list[end] != PosTaging.pos_num:
                            # candidate_spans.append(tuple([bi, end]))
                        candidate_spans.append(tuple([bi, end]))
            cs_num += len(candidate_spans)
            
            target_spans, aspect_spans, opinion_spans = set(), set(), set()
            taos_quads = []
            for quad in quads:
                target_span = quad['target_tags']
                aspect_span = quad['aspect_tags']
                opinion_span = quad['opinion_tags']
                sentiment = quad['sentiment']
                '''obtain span of target, aspect, opinion'''
                if target_span != [-1, -1]:
                    target_span = [tok2id[target_span[0]][0], tok2id[target_span[1]-1][-1]]
                    # target_spans.add(tuple(target_span))
                if aspect_span != [-1, -1]:
                    aspect_span = [tok2id[aspect_span[0]][0], tok2id[aspect_span[1]-1][-1]]
                    # aspect_spans.add(tuple(aspect_span))
                if opinion_span != [-1, -1]:
                    opinion_span = [tok2id[opinion_span[0]][0], tok2id[opinion_span[1]-1][-1]]
                    # opinion_spans.add(tuple(opinion_span))
                
                sent_tag = SentimentTagging._get_tag(sentiment=sentiment)
                q = tuple(target_span) + tuple(aspect_span) + tuple(opinion_span) + tuple([sent_tag])
                taos_quads.append(q)
            
            for t in targets:
                span = [tok2id[t[0]][0], tok2id[t[1]-1][-1]]
                target_spans.add(tuple(span))
            for t in aspects:
                span = [tok2id[t[0]][0], tok2id[t[1]-1][-1]]
                aspect_spans.add(tuple(span))
            for t in opinions:
                span = [tok2id[t[0]][0], tok2id[t[1]-1][-1]]
                opinion_spans.add(tuple(span))
            
            spans = list(target_spans | aspect_spans | opinion_spans)
            target_spans = list(target_spans)
            aspect_spans = list(aspect_spans)
            opinion_spans = list(opinion_spans)
            tar_num += len(target_spans)
            asp_num += len(aspect_spans)
            opi_num += len(opinion_spans)
            maxtar_num = max(maxtar_num, len(target_spans))
            maxasp_num = max(maxasp_num, len(aspect_spans))
            maxopi_num = max(maxopi_num, len(opinion_spans))
            mintar_num = min(mintar_num, len(target_spans))
            minasp_num = min(minasp_num, len(aspect_spans))
            minopi_num = min(minopi_num, len(opinion_spans))
            entities = {
                'targets': target_spans,
                'aspects': aspect_spans,
                'opinions': opinion_spans
            }
            
            all_sss_num += len(spans)
            for sss in spans:
                if sss in candidate_spans:
                    sss_num += 1
                    
            '''set span_tag for target, aspect, opinion spans'''
            span_tags = [SpanTagging.INVALID.value] * len(candidate_spans)
            span_type_tags = [SpanTypeTagging.INVALID.value] * len(candidate_spans)
            for i in range(len(span_tags)):
                if candidate_spans[i] in spans:
                    span_tags[i] = SpanTagging.VALID.value
                if candidate_spans[i] in target_spans:
                    span_type_tags[i] = SpanTypeTagging.TARGET.value
                elif candidate_spans[i] in aspect_spans:
                    span_type_tags[i] = SpanTypeTagging.ASPECT.value
                elif candidate_spans[i] in opinion_spans:
                    span_type_tags[i] = SpanTypeTagging.OPINION.value
            # span_tags = set_tags(spans, token_range, span_tags)
            
            '''reading rank_result file and obtain table label.'''
            k = ranker_result[dialog_id]['k']
            topk_span_indices_idx = torch.LongTensor(ranker_result[dialog_id]['topk_span_indices_idx']).squeeze(0)  # shape: k
            span_indices = torch.LongTensor(candidate_spans)  # shape: span_num, 2
            topk_span_indices = span_indices.gather(dim=0, index=topk_span_indices_idx.unsqueeze(-1).repeat(1, span_indices.size(-1))) # shape: k, 2
            # obtain span_pair_indices
            topk_span_indices = topk_span_indices.unsqueeze(1).expand([-1, k, -1])
            topk_span_indices_T = topk_span_indices.transpose(0, 1)
            span_pair_indices = torch.cat([topk_span_indices, topk_span_indices_T], dim=-1) # shape: k, k, 4
            
            table_labels = set_labels2(span_pair_indices, taos_quads, target_spans, aspect_spans, opinion_spans, SpanTypeTagging, TableTagging)
            
            examples.append(
                InputExample(
                    guid=dialog_id, 
                    dialog=[t for s in sentences for t in s],
                    input_ids=bert_tokens,
                    sentences_num=len(cls_idx_list),
                    sentences_len_list=0,
                    sentence_idx_list=sentence_idx_list,
                    spans=candidate_spans, 
                    span_labels=span_tags, 
                    span_type_labels=span_type_tags,
                    quads=taos_quads,
                    tok2id=tok2id,
                    id2tok=id2tok,
                    span_postag_labels=pos_label_uq_list,
                    ranker_result=[],
                    topk_span_indices_idx=topk_span_indices_idx.tolist(),
                    table_labels=table_labels,
                    span_pair_indices=[],
                    mask_rep=mask_rep,
                    mask_spe=mask_spe,
                    entities=entities
                ))
        
        
        return examples


