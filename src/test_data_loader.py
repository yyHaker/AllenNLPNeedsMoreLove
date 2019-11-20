#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_dataloader.py
@Author  :   yyhaker
@Contact :   572176750@qq.com
@Time    :   2019/11/12 15:41:44
'''

# here put the import lib
import sys
import tempfile
import logging

from data_loader import SquadReaderBert
from models import *

from allennlp.commands.train import train_model
from allennlp.common.params import Params

from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# for pretrained model
tokenizer = PretrainedTransformerTokenizer(
    model_name="bert-base-uncased",
    do_lowercase=True,
    start_tokens=[],
    end_tokens=[]
)

token_indexers = PretrainedTransformerIndexer(
    model_name="bert-base-uncased",
    do_lowercase=True
)

# reader = SquadReader(
#     tokenizer=tokenizer,
#     token_indexers=token_indexers,
#     use_pretrained_model=True
#     )

# reader = SquadReader()

reader = SquadReaderBert(
    tokenizer=tokenizer,
    token_indexers=token_indexers,
    use_pretrained_model=True
)

dev_instances = reader.read("data/squad/dev-v1.1.json")
for idx, instance in enumerate(dev_instances):
    if idx == 10:
        break
    metadata = instance.fields["metadata"].metadata
    print("*" * 200)
    for k, v in metadata.items():
        print(k, " : ", v)

    span_start = instance.fields["span_start"]
    span_end = instance.fields["span_end"]
    print("span_start: ", span_start)
    print("span_end: ", span_end)

    passage_tokens = metadata["context_tokens"]
    print("try to index answer: ", passage_tokens[span_start.sequence_index: span_end.sequence_index + 1])
