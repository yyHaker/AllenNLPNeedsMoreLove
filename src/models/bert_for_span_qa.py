#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bert_for_span_qa.py
@Author  :   yyhaker
@Contact :   572176750@qq.com
@Time    :   2019/11/13 14:45:00
'''

# here put the import lib
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.functional import nll_loss
import os
import random
import traceback
import json

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util, RegularizerApplicator
from allennlp.tools import squad_eval
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

# from pytorch_transformers.modeling_bert import BertModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("BERT_QA")
class BERT_QA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.0,
                 max_span_length: int = 30,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:
        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._max_span_length = max_span_length

        self.qa_outputs = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 2)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._span_qa_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                context: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # the `context` is the concact of `question` and `passage`, so we just use `context`
        batch_size, num_of_passage_tokens = context['tokens'].size()

        # BERT for QA is a fully connected linear layer on top of BERT producing 2 vectors of
        # start and end spans.
        embedded_passage = self._text_field_embedder(context)
        passage_length = embedded_passage.size(1)
        logits = self.qa_outputs(embedded_passage)
        start_logits, end_logits = logits.split(1, dim=-1)
        span_start_logits = start_logits.squeeze(-1)
        span_end_logits = end_logits.squeeze(-1)

        # Adding some masks with numerically stable values
        passage_mask = util.get_text_field_mask(passage).float()
        repeated_passage_mask = passage_mask.unsqueeze(1).repeat(1, 1, 1)
        repeated_passage_mask = repeated_passage_mask.view(batch_size, passage_length)
        span_start_logits = util.replace_masked_values(span_start_logits, repeated_passage_mask, -1e7)
        span_start_probs = util.masked_softmax(span_start_logits, repeated_passage_mask)
        span_end_logits = util.replace_masked_values(span_end_logits, repeated_passage_mask, -1e7)
        span_end_probs = util.masked_softmax(span_end_logits, repeated_passage_mask)
        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict: Dict[str, Any] = {}

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span,
        }

        # compute the loss for training.
        if span_start is not None:
            loss = nll_loss(
                util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1)
            )
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(
                util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1)
            )
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.cat([span_start, span_end], -1))
            output_dict["loss"] = loss

        # Compute the EM and F1 on span qa and add the tokenized input to the output.
        if metadata is not None:
            output_dict["best_span_str"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]["question_tokens"])
                passage_tokens.append(metadata[i]["passage_tokens"])

                passage_words = metadata[i]["paragraph_words"]
                answer_offset = metadata[i]["answer_offset"]
                tok_to_word_index = metadata[i]["tok_to_word_index"]
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_position = tok_to_word_index[predicted_span[0] - answer_offset]
                end_position = tok_to_word_index[predicted_span[1] - answer_offset]
                best_span_str = " ".join(passage_words[start_position: end_position + 1])
                output_dict["best_span_str"].append(best_span_str)
                answer_text = metadata[i].get("answer_text", [])
                if answer_text:
                    answer_text = [answer_text]
                    self._span_qa_metrics(best_span_str, answer_text)
            output_dict["question_tokens"] = question_tokens
            output_dict["passage_tokens"] = passage_tokens

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._span_qa_metrics.get_metric(reset)
        return {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
            "em": exact_match,
            "f1": f1_score,
        }

    @staticmethod
    def get_best_span(
            span_start_logits: torch.Tensor, span_end_logits: torch.Tensor
    ) -> torch.Tensor:
        # We call the inputs "logits" - they could either be unnormalized logits or normalized log
        # probabilities.  A log_softmax operation is a constant shifting of the entire logit
        # vector, so taking an argmax over either one gives the same result.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        span_log_mask = (
            torch.triu(torch.ones((passage_length, passage_length), device=device))
                .log()
                .unsqueeze(0)
        )
        valid_span_log_probs = span_log_probs + span_log_mask

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length
        return torch.stack([span_start_indices, span_end_indices], dim=-1)

