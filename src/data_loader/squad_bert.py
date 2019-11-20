# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   squad_bert.py
@Author  :   yyhaker
@Contact :   572176750@qq.com
@Time    :   2019/11/17 14:25:11
'''

# here put the import lib
import os
import json
import logging
from typing import Any, Dict, List, Tuple, Optional

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer, PretrainedTransformerTokenizer
from allennlp.data.fields import Field, TextField, IndexField, MetadataField

from pytorch_transformers.tokenization_bert import whitespace_tokenize

logger = logging.getLogger(__name__)


@DatasetReader.register("squad_bert")
class SquadReaderBert(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.
    We also support limiting the maximum length for both passage and question. However, some gold
    answer spans may exceed the maximum passage length, which will cause error in making instances.
    We simply skip these spans to avoid errors. If all of the gold answer spans of an example
    are skipped, during training, we will skip this example. During validating or testing, since
    we cannot skip examples, we use the last token as the pseudo gold answer span instead. The
    computed loss will not be accurate as a result. But this will not affect the answer evaluation,
    because we keep all the original gold answer texts.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ``WordTokenizer()``. If use pretrained model, must use `pretrained_transformer`
        ``Tokenizer``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    passage_length_limit : ``int``, optional (default=None)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        if specified, we will cut the question if the length of passage exceeds this limit.
    skip_invalid_examples: ``bool``, optional (default=False)
        if this is true, we will skip those invalid examples
    use_pretrained_model: ``bool``, optional (default=False)
        if this is true, we will use pretrained model method to create ``Instance``.
    max_wordpieces_limit: ``int`` (defualt=384).
        the maximum total input sequence length after WordPiece tokenization. Sequences longer
        than this will be truncated, and sequences shorter than this will be padded.
    """

    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            lazy: bool = True,
            passage_length_limit: int = None,
            question_length_limit: int = None,
            skip_invalid_examples: bool = False,
            use_pretrained_model: bool = False,
            max_wordpieces_limit: int = 384
    ) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples

        self.use_pretrained_model = use_pretrained_model
        self.max_wordpieces_limit = max_wordpieces_limit

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path, cache_dir="data/squad")
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
            logger.info("Reading the dataset")
            for article in dataset[:1]:
                for paragraph_json in article["paragraphs"][:1]:
                    paragraph_text = paragraph_json["context"]
                    # white_space tokenization
                    paragraph_words = []
                    char_to_word_offset = []
                    prev_is_whitespace = True
                    for c in paragraph_text:
                        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                paragraph_words.append(c)
                            else:
                                paragraph_words[-1] += c
                            prev_is_whitespace = False

                        char_to_word_offset.append(len(paragraph_words) - 1)
                    # calc answer span
                    for question_answer in paragraph_json["qas"]:
                        question_id = question_answer["id"]
                        question_text = question_answer["question"].strip().replace("\n", "")
                        # Here, since the givened answer is same, we just choose the first answer just like most does
                        answer = question_answer["answers"][0]
                        answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(paragraph_words[start_position: (end_position + 1)])
                        cleaned_answer_text = " ".join(whitespace_tokenize(answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue
                        # convert to Instance
                        additional_metadata = {"id": question_id}
                        instance = self.text_to_instance(
                            question_text,
                            paragraph_text,
                            paragraph_words,
                            answer_text,
                            start_position,
                            end_position,
                            additional_metadata
                        )
                        if instance is not None:
                            yield instance

    @overrides
    def text_to_instance(
            self,  # type: ignore
            question_text: str,
            paragraph_text: str,
            paragraph_words: List[str],
            answer_text: str = None,
            start_position: int = None,
            end_position: int = None,
            additional_metadata: Dict[str, Any] = None
    ) -> Optional[Instance]:
        # tokenize question
        question_tokens = self._tokenizer.tokenize(question_text)
        if self.question_length_limit is not None:
            question_tokens = question_tokens[0: self.question_length_limit]

        # construct origin(white_space tokenizaton) and token(after word_piece_tokenization) mappings
        tok_to_word_index = []
        word_to_tok_index = []
        paragraph_tokens = []
        for i, word in enumerate(paragraph_words):
            word_to_tok_index.append(len(paragraph_tokens))
            word_pieces = self._tokenizer.tokenize(word)
            for word_piece in word_pieces:
                tok_to_word_index.append(i)
                paragraph_tokens.append(word_piece)

        # limit passage length
        if self.passage_length_limit is not None:
            paragraph_tokens = paragraph_tokens[0: self.passage_length_limit]
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_paragraph = self.max_wordpieces_limit - len(question_tokens) - 3
        paragraph_tokens = paragraph_tokens[0:max_tokens_for_paragraph]

        # concact question and paragraph, called `context`
        # [CLS] + question + [SEP] + paragraph + [SEP]
        context_tokens = [Token("[cls]")] + question_tokens + [Token("[sep]")] + paragraph_tokens + [Token("[sep]")]
        answer_offset = len(question_tokens) + 2

        # calc answer span
        tok_start_position = None
        tok_end_position = None
        if start_position and end_position:
            tok_start_position = word_to_tok_index[start_position]
            tok_end_position = word_to_tok_index[end_position]
            if self.passage_length_limit is not None and tok_end_position > self.passage_length_limit:
                return None
            (tok_start_position, tok_end_position) = self._improve_answer_span(
                paragraph_tokens, tok_start_position, tok_end_position, self._tokenizer, answer_text
            )
            tok_start_position += answer_offset
            tok_end_position += answer_offset

        # for easy to calc predict answer
        additional_metadata = additional_metadata or {}
        additional_metadata["answer_offset"] = answer_offset
        additional_metadata["tok_to_word_index"] = tok_to_word_index
        additional_metadata["word_to_tok_index"] = word_to_tok_index
        additional_metadata["paragraph_words"] = paragraph_words

        return self._make_qa_instance(
            question_tokens,
            paragraph_tokens,
            context_tokens,
            question_text,
            paragraph_text,
            answer_text,
            tok_start_position,
            tok_end_position,
            additional_metadata
        )

    def _make_qa_instance(
            self,  # type: ignore
            question_tokens: List[Token],
            passage_tokens: List[Token],
            context_tokens: List[Token],
            question_text: str,
            passage_text: str,
            answer_text: str = None,
            start_position: int = None,
            end_position: int = None,
            additional_metadata: Dict[str, Any] = None
    ) -> Instance:
        """
        Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
        in a reading comprehension model.
        """
        additional_metadata = additional_metadata or {}
        fileds: Dict[str, Field] = {}
        fileds["question"] = TextField(question_tokens, self._token_indexers)
        fileds["passage"] = TextField(passage_tokens, self._token_indexers)
        context_filed = TextField(context_tokens, self._token_indexers)
        fileds["context"] = context_tokens
        metadata = {
            'question_tokens': [token.text for token in question_tokens],
            'passage_tokens': [token.text for token in passage_tokens],
            'context_tokens': [token.text for token in context_tokens],
            'question_text': question_text,
            'passage_text': passage_text
        }
        if answer_text:
            metadata["answer_text"] = answer_text
        if start_position and end_position:
            fileds["span_start"] = IndexField(start_position, context_filed)
            fileds["span_end"] = IndexField(start_position, context_filed)

        metadata.update(additional_metadata)
        fileds["metadata"] = MetadataField(metadata)
        return Instance(fileds)

    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""
        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join([token.text for token in tokenizer.tokenize(orig_answer_text)])
        doc_tokens = [token.text for token in doc_tokens]

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)
