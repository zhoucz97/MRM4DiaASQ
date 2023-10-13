#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import abc
import six
import torch


def metrics(probability, labels):
    epsilon = 1e-6
    num_correct = torch.logical_and(labels == probability.argmax(-1), probability.argmax(-1) != 0).sum().item()
    num_proposed = (probability.argmax(-1) != 0).sum().item()
    num_gold = (labels != 0).sum().item()
    # accuracy
    # num_all_correct = (labels == probability.argmax(-1)).sum().item()
    # num_total = probability.size(0) * probability.size(1)

    precision = num_correct / (num_proposed + epsilon)
    recall = num_correct / (num_gold + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    # accuracy = num_all_correct / (num_total + epsilon)
    return precision, recall, f1


@six.add_metaclass(abc.ABCMeta)
class Metric(object):

    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset states and result
        """
        raise NotImplementedError("function 'reset' not implemented in {}.".
                                  format(self.__class__.__name__))

    @abc.abstractmethod
    def update(self, *args):
        """
        Update states for metric

        Inputs of :code:`update` is the outputs of :code:`Metric.compute`,
        if :code:`compute` is not defined, the inputs of :code:`update`
        will be flatten arguments of **output** of mode and **label** from data:
        :code:`update(output1, output2, ..., label1, label2,...)`

        see :code:`Metric.compute`
        """
        raise NotImplementedError("function 'update' not implemented in {}.".
                                  format(self.__class__.__name__))

    @abc.abstractmethod
    def accumulate(self):
        """
        Accumulates statistics, computes and returns the metric value
        """
        raise NotImplementedError(
            "function 'accumulate' not implemented in {}.".format(
                self.__class__.__name__))

    @abc.abstractmethod
    def name(self):
        """
        Returns metric name
        """
        raise NotImplementedError("function 'name' not implemented in {}.".
                                  format(self.__class__.__name__))

    def compute(self, *args):
        """
        This API is advanced usage to accelerate metric calculating, calulations
        from outputs of model to the states which should be updated by Metric can
        be defined here, where Paddle OPs is also supported. Outputs of this API
        will be the inputs of "Metric.update".

        If :code:`compute` is defined, it will be called with **outputs**
        of model and **labels** from data as arguments, all outputs and labels
        will be concatenated and flatten and each filed as a separate argument
        as follows:
        :code:`compute(output1, output2, ..., label1, label2,...)`

        If :code:`compute` is not defined, default behaviour is to pass
        input to output, so output format will be:
        :code:`return output1, output2, ..., label1, label2,...`

        see :code:`Metric.update`
        """
        return args


class SpanEvaluator(Metric):
    """
    SpanEvaluator computes the precision, recall and F1-score for span detection.
    """

    def __init__(self):
        super(SpanEvaluator, self).__init__()
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def compute(self, probability, labels):
        """
        Computes the precision, recall and F1-score for span detection.
        """
        num_correct_spans = 0
        num_infer_spans = 0
        num_label_spans = 0

        num_correct_spans = torch.logical_and(labels == probability.argmax(-1), probability.argmax(-1) != 0).sum().item()
        num_infer_spans = (probability.argmax(-1) != 0).sum().item()
        num_label_spans = (labels != 0).sum().item()



        return num_correct_spans, num_infer_spans, num_label_spans

    def update(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_spans += num_infer_spans
        self.num_label_spans += num_label_spans
        self.num_correct_spans += num_correct_spans

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """
        precision = float(self.num_correct_spans /
                          self.num_infer_spans) if self.num_infer_spans else 0.
        recall = float(self.num_correct_spans /
                       self.num_label_spans) if self.num_label_spans else 0.
        f1_score = float(2 * precision * recall /
                         (precision + recall)) if self.num_correct_spans else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"


class QuadEvaluator(Metric):
    """
    QuadEvaluator computes the precision, recall and F1-score for span detection.
    """
    def __init__(self):
        super(QuadEvaluator, self).__init__()
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def compute(self, probability, labels):
        """
        Computes the precision, recall and F1-score for span detection.
        """
        num_correct_spans = 0
        num_infer_spans = 0
        num_label_spans = 0

        num_correct_spans = torch.logical_and(labels == probability.argmax(-1), probability.argmax(-1) != 0).sum().item()
        num_infer_spans = (probability.argmax(-1) != 0).sum().item()
        num_label_spans = (labels != 0).sum().item()

        return num_correct_spans, num_infer_spans, num_label_spans

    def update(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_spans += num_infer_spans
        self.num_label_spans += num_label_spans
        self.num_correct_spans += num_correct_spans

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """
        precision = float(self.num_correct_spans /
                          self.num_infer_spans) if self.num_infer_spans else 0.
        recall = float(self.num_correct_spans /
                       self.num_label_spans) if self.num_label_spans else 0.
        f1_score = float(2 * precision * recall /
                         (precision + recall)) if self.num_correct_spans else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"
