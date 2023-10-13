#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from enum import IntEnum
from typing import Tuple, List, Text
from pydantic import BaseModel
import nltk


class SpanTagging(IntEnum):
    INVALID = 0
    VALID = 1

class SpanTypeTagging(IntEnum):
    INVALID = 0
    TARGET = 1
    ASPECT = 2
    OPINION = 3

class TableTagging(IntEnum):
    INVALID = 0
    TARGET = 1
    ASPECT = 2
    OPINION = 3
    TA = 4
    AO = 5
    TO_POS = 6
    TO_NEG = 7
    TO_NEU = 8
    
class SentimentTagging(IntEnum):
    POSITIVE = 0
    NEGATIVE = 1
    OTHER = 2
    
    @classmethod
    def _get_tag(self, sentiment):
        if sentiment == 'positive' or sentiment == 'pos':
            return self.POSITIVE.value
        elif sentiment == 'negative' or sentiment == 'neg':
            return self.NEGATIVE.value
        else:
            return self.OTHER.value
        
    @classmethod
    def _get_sent(self, idx):
        if idx == 0:
            return "pos"
        elif idx == 1:
            return "neg"
        else:
            return "other"
        
    
    
class ZHPosTaging():
    pos_list = ['BERT_TOKEN', 'a', 'ad', 'b', 'c', 'd', 'df', 'e', 'eng', 'f', 
                'i', 'j', 'k', 'l', 'm', 'mq', 'n', 'ng', 'nr', 'nrt', 'ns', 'nz', 
                'o', 'p', 'q', 'r', 's', 't', 'u', 'ud', 'ug', 'uj', 'ul', 'uz', 
                'v', 'vg', 'vn', 'x', 'y', 'z', 'zg']

    pos_num = len(pos_list) # add unk token index

    @classmethod
    def get_label(self, pt: str):
        try:
            return self.pos_list.index(pt)
        except ValueError:
            return self.pos_num

    @classmethod
    def get_sentence_posTag(self, pos_tagger, sentence: List):
        seg_w_list, seg_tag_list = [], []
        for w, t in pos_tagger.cut(''.join(sentence[1:-1])):
            seg_w_list.append(w)
            seg_tag_list.extend([t] * len(w))
        
        seg_tag_list2 = ['BERT_TOKEN']
        i = 0
        for w in sentence[1:-1]:
            if len(w) == 1:
                seg_tag_list2.append(seg_tag_list[i])
            else:
                from collections import Counter
                t_tag = Counter(seg_tag_list[i: i+len(w)]).most_common(1)[0][0]
                seg_tag_list2.append(t_tag)
            i += len(w)
        seg_tag_list2.append('BERT_TOKEN')
        assert len(sentence) == len(seg_tag_list2)
            
        return seg_tag_list2

    @classmethod
    def get_word_posTag(self, pos_tagger, word: str):
        pos_tag = pos_tagger([word])[0][1]
        return pos_tag


class PosTaging():

    all_pos_list = ['DT', 'NN', 'VBZ', 'RB', 'JJ', ',', 'CC', 'VBN', 'TO', 'PRP', 'VBP', 
     'RBR', 'IN', 'NNP', "''", 'NNS', '.', 'VB', ':', 'EX', 'CD', 'WP', 'MD', 'VBG', 
     'PRP$', 'POS', 'VBD', 'WRB', '$', 'JJR', 'RBS', 'JJS', 'RP', 'WDT', '(', ')', 'FW', 
     'PDT', 'UH', 'NNPS', 'SYM', '#', 'WP$']

    pos_list     = ['DT', 'NN', 'VBZ', 'RB', 'JJ',      'CC', 'VBN', 'TO', 'PRP', 'VBP', 
     'RBR', 'IN', 'NNP',       'NNS',      'VB',      'EX', 'CD', 'WP', 'MD', 'VBG', 
     'PRP$', 'POS', 'VBD', 'WRB',      'JJR', 'RBS', 'JJS', 'RP', 'WDT',           'FW', 
     'PDT', 'UH', 'NNPS',                  ]
    pos_num = len(pos_list) # add unk token index
    

    @classmethod
    def get_label(self, pt: str):
        try:
            return self.pos_list.index(pt)
        except ValueError:
            return self.pos_num

    @classmethod
    def get_sentence_posTag(self, pos_tagger, sentence: List):
        pos_tag = pos_tagger(sentence)
        pos_tag = [pt[1] for pt in pos_tag]
        return pos_tag

    @classmethod
    def get_word_posTag(self, pos_tagger, word: str):
        pos_tag = pos_tagger([word])[0][1]
        return pos_tag


class SentimentQuad(BaseModel):
    target: List
    aspect: List
    opinion: List
    sentiment: Text

    @classmethod
    def from_sentiment_quad(cls, labels: Tuple[List, List, List, Text]):
        """read from sentiment quad"""
        assert len(labels) == 4  # target, aspect, opinion, sentiment
        # sentiments: {'neu', 'neg', 'pos', 'doubt', 'amb', -1}
        if labels[3] == 'pos':
            sentiment = 'pos'
        elif labels[3] == 'neg':
            sentiment = 'neg'
        else:
            sentiment = 'other'
        return cls(
            target=labels[0],
            aspect=labels[1],
            opinion=labels[2],
            sentiment=sentiment
        )