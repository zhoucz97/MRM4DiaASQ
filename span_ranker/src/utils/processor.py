#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy
import json
import os
from enum import IntEnum
import nltk
import jieba.posseg as psg

from utils.tager import SpanTagging, SpanTypeTagging, TableTagging, PosTaging, ZHPosTaging, SentimentTagging


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, dialog, input_ids, sentences_num, sentences_len_list, 
                 spans, span_labels, span_type_labels, quads, tok2id, id2tok, span_postag_labels):
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
    def __init__(self, tokenizer, max_length, span_max_len):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.span_max_len = span_max_len
        self.pos_tagger = nltk.pos_tag

    def get_examples(self, data_path, data_type):
        """See base class."""
        return self._create_examples(data_path, data_type)
    
    def _add_spec_token_idx(self, span, conv_lens):
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
            quads = []
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
            for i, t in enumerate(triplets):
                t1, t2, a1, a2, o1, o2, sentiment = t[0:7]
                nt1, nt2 = self._add_spec_token_idx([t1, t2], orig_sent_lens)
                na1, na2 = self._add_spec_token_idx([a1, a2], orig_sent_lens)
                no1, no2 = self._add_spec_token_idx([o1, o2], orig_sent_lens)
                target_tags, t_sent_idx = self._obtain_tags(sent_list, nt1, nt2)
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
        """Creates examples for the data sets."""
        cs_num = 0
        trun_sss_num, true_sss_num = 0, 0
        tag_set = list()
        
        token_sum = 0
        
        dialogs = self._read_orig_files(data_dir, data_type)
        
        examples = []
        for dialog in dialogs:
            dialog_id = dialog['id']
            sentences = dialog['sentence']
            if 'zh' in data_dir:
                sentences = [[w.strip() for w in sent] for sent in sentences]
            
            quads = dialog['quads']
            targets = dialog['entities']['targets']
            aspects = dialog['entities']['aspects']
            opinions = dialog['entities']['opinions']
            
            dialog_len = len([s for sents in sentences for s in sents])
            pad_idx = 1
            
            bert_tokens = []
            bert_tokens_padding = [pad_idx] * self.max_length
            
            # set pos tag for each word
            pos_tag_list, pos_label_list = [], []
            if 'en' in data_type:  # english dataset 
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
                    tag_set.extend(ZHPosTaging.get_sentence_posTag(self.pos_tagger, sent))
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
            assert len(is_suffix_list) == len(bert_tokens)
            assert len(pos_label_uq_list) == len(bert_tokens)
            assert len(tok2id) == dialog_len
            assert len(id2tok) == len(bert_tokens)
            
            
            candidate_spans = []
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
            
                 
            bert_len = len(bert_tokens)
            for i in range(bert_len):
                bert_tokens_padding[i] = bert_tokens[i]

            target_spans, aspect_spans, opinion_spans = set(), set(), set()
            taos_quads = []
            for quad in quads:
                target_span = quad['target_tags']
                aspect_span = quad['aspect_tags']
                opinion_span = quad['opinion_tags']
                sentiment = quad['sentiment']

                target_span = [tok2id[target_span[0]][0], tok2id[target_span[1]-1][-1]]
                aspect_span = [tok2id[aspect_span[0]][0], tok2id[aspect_span[1]-1][-1]]
                opinion_span = [tok2id[opinion_span[0]][0], tok2id[opinion_span[1]-1][-1]]

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
            
            true_sss_num += len(spans)
            for sss in spans:
                if sss in candidate_spans:
                    trun_sss_num += 1
                    

            '''set span tag for target, aspect, opinion spans'''
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
            
            token_sum += len(bert_tokens)
            '''set pair tag for table'''
            examples.append(
                InputExample(
                    guid=dialog_id, 
                    dialog=[t for s in sentences for t in s],
                    input_ids=bert_tokens,
                    sentences_num=len(cls_idx_list),
                    sentences_len_list=0,
                    spans=candidate_spans, 
                    span_labels=span_tags, 
                    span_type_labels=span_type_tags,
                    quads=taos_quads,
                    tok2id=tok2id,
                    id2tok=id2tok,
                    span_postag_labels=pos_label_uq_list
                ))
        
        return examples
    
    




