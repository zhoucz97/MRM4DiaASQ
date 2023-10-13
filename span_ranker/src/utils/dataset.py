#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from typing import Text
from torch.utils.data import Dataset
from utils.processor import InputExample, QuadDataProcessor
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    """
    An customer class representing txt data reading
    """
    def __init__(self,
                 data_type: "Text",
                 data_path: "Text",
                 processor: "QuadDataProcessor",
                 tokenizer: "AutoTokenizer",
                 max_seq_length: "int"
                 ) -> "None":
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.sentence_list = []
        self.examples = processor.get_examples(data_path, data_type)
        # self.examples = examples

    def __getitem__(self, idx: "int"):
        example = self.examples[idx]  # type:InputExample
        """Example:
        guid, dialog, input_ids, sentences_num, sentences_len_list, 
        spans, span_labels, span_type_labels, quads, tok2id, id2tok, span_postag_labels
        """
        guid = example.guid
        input_ids = example.input_ids
        spans = example.spans
        span_labels = example.span_labels
        span_type_labels = example.span_type_labels
        span_postag_labels = example.span_postag_labels
        tok2id = example.tok2id
        id2tok = example.id2tok

        return guid, tok2id, id2tok, input_ids, spans, span_labels, span_type_labels, span_postag_labels

    def __len__(self):
        return len(self.examples)
