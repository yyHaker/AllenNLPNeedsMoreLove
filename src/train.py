#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Author  :   yyhaker
@Contact :   572176750@qq.com
@Time    :   2019/11/05 16:41:35
'''

# here put the import lib
import sys
sys.path.append("src")
import tempfile

from data_loader import *
from models import *

from allennlp.commands.train import train_model
from allennlp.common.params import Params

params = Params.from_file("config_file/snli_bert.jsonnet")
serialization_dir = tempfile.mkdtemp()
model = train_model(params, serialization_dir, force=True, cache_directory="temp/snli/bert")

