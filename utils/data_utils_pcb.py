# -*- coding:utf-8 -*
import logging
import os
import json
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, sl_words, sl_labels_ed, sl_order, tl_words, tl_labels_ed, tl_order):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.sl_words = sl_words
        self.sl_labels_ed = sl_labels_ed
        self.sl_order = sl_order
        self.tl_words = tl_words
        self.tl_labels_ed = tl_labels_ed
        self.tl_order = tl_order


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_sl, input_ids_tl, input_mask_sl, input_mask_tl, \
        sl_label_ids, tl_label_ids, sl_label_mask, tl_label_mask, \
        sl_order, tl_order):
        
        self.input_ids_sl = input_ids_sl
        self.input_mask_sl = input_mask_sl
        self.sl_label_ids = sl_label_ids
        self.sl_label_mask = sl_label_mask
        self.sl_order = sl_order

        self.input_ids_tl = input_ids_tl
        self.input_mask_tl = input_mask_tl
        self.tl_label_ids = tl_label_ids
        self.tl_label_mask = tl_label_mask
        self.tl_order = tl_order

def read_examples_from_file(args, data_dir, mode):
    # file_path = os.path.join(data_dir, "{}.{}.finalall.json".format(args.dataset, mode)) #de.train.finalall
    guid_index = 1
    examples = []

    # with open(file_path, 'r') as f:
    #     data = json.load(f)
    #     for item in data:
    #         sl_words = item["str_words_sl"]
    #         # labels_ner = item["tags_ner"]
    #         sl_labels_ed = item["tags_ed_sl"]
    #         # labels_net = item["tags_net"]
    #         sl_order = item["trans_order_sl"]
    #         tl_words = item["str_words_tl"]
    #         tl_labels_ed = item["tags_ed_tl"]
    #         tl_order = item["trans_order_tl"]
    #         examples.append(InputExample(guid="%s-%d".format(mode, guid_index), sl_words=sl_words, sl_labels_ed=sl_labels_ed, sl_order=sl_order, \
    #                           tl_words=tl_words, tl_labels_ed=tl_labels_ed, tl_order=tl_order))
    #         guid_index += 1
    
    candidate_lst = ["sst", "stts", "tst"]

    for ii in candidate_lst:

        file_path = os.path.join(data_dir, "{}.{}.{}.json".format(args.dataset, mode, ii)) #de.train.sst.json

        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                sl_words = item["str_words_sl"]
                sl_labels_ed = item["tags_ed_sl"]
                sl_order = item["trans_order_sl"]
                tl_words = item["str_words_tl"]
                tl_labels_ed = item["tags_ed_tl"]
                tl_order = item["trans_order_tl"]
                examples.append(InputExample(guid="%s-%d".format(mode, guid_index), sl_words=sl_words, sl_labels_ed=sl_labels_ed, sl_order=sl_order, \
                                  tl_words=tl_words, tl_labels_ed=tl_labels_ed, tl_order=tl_order))
                guid_index += 1

    return examples

def convert_mixed_examples_to_features(
    tag_to_id,
    examples,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum = -1,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        Mixed Language: [CLS] s1, s2, ... [SEP] t1, t2, ... [SEP]
                    --> [CLS] s1, ..., [MASK], ... [SEP] t1, t2, ... [SEP]
                    <-- [CLS] s1, s2, ... [SEP] t1, ..., <s> ti, ... <t>, ... [SEP]
        Tag Mask: source: 0, target: 1, [MASK]: 2, <s> ti, ... <t>: 3, PADDING: -1
    """ 
    features = []
    extra_long_samples = 0
    # span_non_id = tag_to_id["span"]["O"]
    # type_non_id = tag_to_id["type"]["O"]
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        sl_tokens = []
        tl_tokens = []
        sl_label_ids = []
        tl_label_ids = []
        sl_label_mask = []
        tl_label_mask = []
        sl_order = []
        tl_order = []
        # print(len(example.words), len(example.labels))
        for sl_word, sl_label, sl_order_ in zip(example.sl_words, example.sl_labels_ed, example.sl_order):
            sl_label = sl_label.split("-")[0]
            # print(word, label)
            sl_label = tag_to_id["span"][sl_label]
            # tl_label = tag_to_id["span"][tl_label]
            # type_label = tag_to_id["type"][type_label]
            sl_word_tokens = tokenizer.tokenize(sl_word)
            if len(sl_word_tokens) > 0:
                sl_tokens.extend(sl_word_tokens)
                sl_label_ids.extend([sl_label] + [pad_token_label_id] * (len(sl_word_tokens) - 1))
                sl_label_mask.extend([1] + [0]*(len(sl_word_tokens) - 1))
                sl_order.extend([sl_order_] + [pad_token_label_id] * (len(sl_word_tokens) - 1))

        for tl_word, tl_label, tl_order_ in zip(example.tl_words, example.tl_labels_ed, example.tl_order):
            tl_label = tl_label.split("-")[0]
            # print(word, label)
            # sl_label = tag_to_id["span"][sl_label]
            tl_label = tag_to_id["span"][tl_label]
            # type_label = tag_to_id["type"][type_label]
            tl_word_tokens = tokenizer.tokenize(tl_word)
            if len(tl_word_tokens) > 0:
                tl_tokens.extend(tl_word_tokens)
                tl_label_ids.extend([tl_label] + [pad_token_label_id] * (len(tl_word_tokens) - 1))
                tl_label_mask.extend([1] + [0]*(len(tl_word_tokens) - 1))
                tl_order.extend([tl_order_] + [pad_token_label_id] * (len(tl_word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(sl_tokens) > max_seq_length - special_tokens_count:
            sl_tokens = sl_tokens[: (max_seq_length - special_tokens_count)]
            sl_label_ids = sl_label_ids[: (max_seq_length - special_tokens_count)]
            sl_label_mask = sl_label_mask[: (max_seq_length - special_tokens_count)]
            sl_order = sl_order[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1
        if len(tl_tokens) > max_seq_length - special_tokens_count:
            tl_tokens = tl_tokens[: (max_seq_length - special_tokens_count)]
            tl_label_ids = tl_label_ids[: (max_seq_length - special_tokens_count)]
            tl_label_mask = tl_label_mask[: (max_seq_length - special_tokens_count)]
            tl_order = tl_order[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1
        
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        sl_tokens += [sep_token]
        sl_label_ids += [pad_token_label_id]
        sl_label_mask += [1]
        sl_order += [pad_token_label_id]

        tl_tokens += [sep_token]
        tl_label_ids += [pad_token_label_id]
        tl_label_mask += [1]
        tl_order += [pad_token_label_id]
        # type_label_ids += [pad_token_label_id]
        # label_mask += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            sl_tokens += [sep_token]
            sl_label_ids += [pad_token_label_id]
            sl_label_mask += [1]
            sl_order += [pad_token_label_id]

            tl_tokens += [sep_token]
            tl_label_ids += [pad_token_label_id]
            tl_label_mask += [1]
            tl_order += [pad_token_label_id]

        # segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            sl_tokens += [cls_token]
            sl_label_ids += [pad_token_label_id]
            sl_label_mask += [1]
            sl_order += [pad_token_label_id]

            tl_tokens += [cls_token]
            tl_label_ids += [pad_token_label_id]
            tl_label_mask += [1]
            tl_order += [pad_token_label_id]
        else:
            sl_tokens = [cls_token] + sl_tokens
            sl_label_ids = [pad_token_label_id] + sl_label_ids
            sl_label_mask = [1] + sl_label_mask
            sl_order  = [pad_token_label_id] + sl_order

            tl_tokens = [cls_token] + tl_tokens
            tl_label_ids = [pad_token_label_id] + tl_label_ids
            tl_label_mask = [1] + tl_label_mask
            tl_order  = [pad_token_label_id] + tl_order

        input_ids_sl = tokenizer.convert_tokens_to_ids(sl_tokens)
        input_ids_tl = tokenizer.convert_tokens_to_ids(tl_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask_sl = [1 if mask_padding_with_zero else 0] * len(input_ids_sl)
        input_mask_tl = [1 if mask_padding_with_zero else 0] * len(input_ids_tl)

        # Zero-pad up to the sequence length.
        padding_length_sl = max_seq_length - len(input_ids_sl)
        padding_length_tl = max_seq_length - len(input_ids_tl)
        if pad_on_left:
            input_ids_sl = ([pad_token] * padding_length_sl) + input_ids_sl
            input_mask_sl = ([0 if mask_padding_with_zero else 1] * padding_length_sl) + input_mask_sl
            sl_label_ids = ([pad_token_label_id] * padding_length_sl) + sl_label_ids
            sl_label_mask = ([-1] * padding_length_sl) + sl_label_mask
            sl_order = ([pad_token_label_id] * padding_length_sl) + sl_order

            input_ids_tl = ([pad_token] * padding_length_tl) + input_ids_tl
            input_mask_tl = ([0 if mask_padding_with_zero else 1] * padding_length_tl) + input_mask_tl
            tl_label_ids = ([pad_token_label_id] * padding_length_tl) + tl_label_ids
            tl_label_mask = ([-1] * padding_length_tl) + tl_label_mask
            tl_order = ([pad_token_label_id] * padding_length_tl) + tl_order
        else:
            input_ids_sl += [pad_token] * padding_length_sl
            input_mask_sl += [0 if mask_padding_with_zero else 1] * padding_length_sl
            sl_label_ids += [pad_token_label_id] * padding_length_sl
            sl_label_mask += [-1] * padding_length_sl
            sl_order += [pad_token_label_id] * padding_length_sl

            input_ids_tl += [pad_token] * padding_length_tl
            input_mask_tl += [0 if mask_padding_with_zero else 1] * padding_length_tl
            tl_label_ids += [pad_token_label_id] * padding_length_tl
            tl_label_mask += [-1] * padding_length_tl
            tl_order += [pad_token_label_id] * padding_length_tl
        
        # print(len(input_ids))
        # print(len(label_ids))
        # print(max_seq_length)
        assert len(input_ids_sl) == max_seq_length
        assert len(input_mask_sl) == max_seq_length
        assert len(sl_label_ids) == max_seq_length
        assert len(sl_label_mask) == max_seq_length
        assert len(sl_order) == max_seq_length

        assert len(input_ids_tl) == max_seq_length
        assert len(input_mask_tl) == max_seq_length
        assert len(tl_label_ids) == max_seq_length
        assert len(tl_label_mask) == max_seq_length
        assert len(tl_order) == max_seq_length

        if ex_index < show_exnum:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("sl_tokens: %s", " ".join([str(x) for x in sl_tokens]))
            logger.info("input_ids_sl: %s", " ".join([str(x) for x in input_ids_sl]))
            logger.info("input_mask_sl: %s", " ".join([str(x) for x in input_mask_sl]))
            logger.info("sl_label_ids: %s", " ".join([str(x) for x in sl_label_ids]))
            logger.info("sl_label_mask: %s", " ".join([str(x) for x in sl_label_mask]))
            logger.info("sl_order: %s", " ".join([str(x) for x in sl_order]))

            logger.info("tl_tokens: %s", " ".join([str(x) for x in tl_tokens]))
            logger.info("input_ids_tl: %s", " ".join([str(x) for x in input_ids_tl]))
            logger.info("input_mask_tl: %s", " ".join([str(x) for x in input_mask_tl]))
            logger.info("tl_label_ids: %s", " ".join([str(x) for x in tl_label_ids]))
            logger.info("tl_label_mask: %s", " ".join([str(x) for x in tl_label_mask]))
            logger.info("tl_order: %s", " ".join([str(x) for x in tl_order]))
        # input_ids, input_mask, segment_ids, label_ids, full_label_ids, span_label_ids, type_label_ids
        features.append(
            InputFeatures(input_ids_sl=input_ids_sl, input_ids_tl=input_ids_tl, \
                input_mask_sl=input_mask_sl, input_mask_tl=input_mask_tl, \
                sl_label_ids=sl_label_ids, tl_label_ids=tl_label_ids, \
                sl_label_mask=sl_label_mask, tl_label_mask=tl_label_mask, \
                sl_order=sl_order, tl_order=tl_order)
        )
    logger.info("Extra long example %d of %d", extra_long_samples, len(examples))
    
    return features


def convert_examples_to_features(
    tag_to_id,
    examples,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum = -1,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    extra_long_samples = 0
    span_non_id = tag_to_id["span"]["O"]
    type_non_id = tag_to_id["type"]["O"]
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        span_label_ids = []
        type_label_ids = []
        label_mask = []
        # print(len(example.words), len(example.labels))
        for word, span_label, type_label in zip(example.words, example.spans, example.types):
            # print(word, label)
            span_label = tag_to_id["span"][span_label]
            type_label = tag_to_id["type"][type_label]
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if len(word_tokens) > 0:
                span_label_ids.extend([span_label] + [pad_token_label_id] * (len(word_tokens) - 1))
                type_label_ids.extend([type_label] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_mask.extend([1] + [0]*(len(word_tokens) - 1))
            # full_label_ids.extend([label] * len(word_tokens))

        # print(len(tokens), len(label_ids), len(full_label_ids))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            span_label_ids = span_label_ids[: (max_seq_length - special_tokens_count)]
            type_label_ids = type_label_ids[: (max_seq_length - special_tokens_count)]
            label_mask = label_mask[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        span_label_ids += [pad_token_label_id]
        type_label_ids += [pad_token_label_id]
        label_mask += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            span_label_ids += [pad_token_label_id]
            type_label_ids += [pad_token_label_id]
            label_mask += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            span_label_ids += [pad_token_label_id]
            type_label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            label_mask += [0]
        else:
            tokens = [cls_token] + tokens
            span_label_ids = [pad_token_label_id] + span_label_ids
            type_label_ids = [pad_token_label_id] + type_label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            label_mask += [0]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            span_label_ids = ([pad_token_label_id] * padding_length) + span_label_ids
            type_label_ids = ([pad_token_label_id] * padding_length) + type_label_ids
            label_mask = ([0] * padding_length) + label_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            span_label_ids += [pad_token_label_id] * padding_length
            type_label_ids += [pad_token_label_id] * padding_length
            label_mask += [0] * padding_length
        
        # print(len(input_ids))
        # print(len(label_ids))
        # print(max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(span_label_ids) == max_seq_length
        assert len(type_label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < show_exnum:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("span_label_ids: %s", " ".join([str(x) for x in span_label_ids]))
            logger.info("type_label_ids: %s", " ".join([str(x) for x in type_label_ids]))
            logger.info("label_mask: %s", " ".join([str(x) for x in label_mask]))
        # input_ids, input_mask, segment_ids, label_ids, full_label_ids, span_label_ids, type_label_ids
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, \
                label_ids=None, full_label_ids=None, span_label_ids=span_label_ids, type_label_ids=type_label_ids, label_mask=label_mask)
        )
    logger.info("Extra long example %d of %d", extra_long_samples, len(examples))
    
    return features

def load_and_cache_examples(args, tokenizer, pad_token_label_id, mode):

    tags_to_id = tag_to_id(args.data_dir, args.dataset)
    # tags_to_id_src = tag_to_id(args.data_dir, "twitter")
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "{}_{}.pt".format(
            args.dataset, mode
        ),
    )

    # cached_features_file_src = None

    # if mode == "train":
    #     cached_features_file_src = os.path.join(
    #         args.data_dir,
    #         "{}_{}.pt".format(
    #             args.src_dataset, mode
    #         ),
    #     )


    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        # if mode == "train":
        #     logger.info("Loading source domain features from cached file %s", cached_features_file_src)
        #     features_src = torch.load(cached_features_file_src)
        #     # features_inter = torch.load(cached_features_file_inter)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args, args.data_dir, mode)
        features = convert_mixed_examples_to_features(
            tags_to_id,
            examples,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end = bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token = tokenizer.cls_token,
            cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0,
            sep_token = tokenizer.sep_token,
            sep_token_extra = bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left = bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id = pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            # if mode == "train":
            #     logger.info("Saving features into cached file %s", cached_features_file_src)
            #     torch.save(features_src, cached_features_file_src)
            #     # logger.info("Saving features into cached file %s", cached_features_file_inter)
            #     # torch.save(features_inter, cached_features_file_inter)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids_sl = torch.tensor([f.input_ids_sl for f in features], dtype=torch.long)
    all_input_ids_tl = torch.tensor([f.input_ids_tl for f in features], dtype=torch.long)
    all_input_mask_sl = torch.tensor([f.input_mask_sl for f in features], dtype=torch.long)
    all_input_mask_tl = torch.tensor([f.input_mask_tl for f in features], dtype=torch.long)
    all_label_ids_sl = torch.tensor([f.sl_label_ids for f in features], dtype=torch.long)
    all_label_ids_tl = torch.tensor([f.tl_label_ids for f in features], dtype=torch.long)
    all_label_mask_sl = torch.tensor([f.sl_label_mask for f in features], dtype=torch.long)
    all_label_mask_tl = torch.tensor([f.tl_label_mask for f in features], dtype=torch.long)
    all_order_sl = torch.tensor([f.sl_order for f in features], dtype=torch.long)
    all_order_tl = torch.tensor([f.tl_order for f in features], dtype=torch.long)
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)

    dataset = TensorDataset(all_input_ids_sl, all_input_ids_tl, \
        all_input_mask_sl, all_input_mask_tl, \
        all_label_ids_sl, all_label_ids_tl, \
        all_label_mask_sl, all_label_mask_tl, \
        all_order_sl, all_order_tl \
    )

    return dataset

def get_labels(path=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        labels_ner = {}
        labels_span = {}
        labels_type = {}
        non_entity_id = None
        with open(path+dataset+"_tag_to_id.json", "r") as f:
            data = json.load(f)
            spans = data["span"]
            for l, idx in spans.items():
                labels_span[idx] = l
            types = data["type"]
            for l, idx in types.items():
                labels_type[idx] = l

        # labels_type_src = {}
        # with open(path+"twitter_tag_to_id.json", "r") as f:
        #     data = json.load(f)
        #     # spans = data["span"]
        #     # for l, idx in spans.items():
        #     #     labels_span[idx] = l
        #     types = data["type"]
        #     for l, idx in types.items():
        #         labels_type_src[idx] = l

        # if "O" not in labels:
        #     labels = ["O"] + labels
        # print(labels_type_src)
        return labels_span, labels_type
    else:
        return None, None, None

def tag_to_id(path=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        with open(path+dataset+"_tag_to_id.json", 'r') as f:
            data = json.load(f)
        return data # {"ner":{}, "span":{}, "type":{}}
    else:
        return None

def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq_type, seq_bio, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    assert len(seq_bio) == len(seq_type)
    spans = tags["span"]
    default = spans["O"]
    bgn = spans["B"]
    inner = spans["I"]
    idx_to_tag = {idx: tag for tag, idx in spans.items()}
    types = tags["type"]
    idx_to_type = {idx: t for t, idx in types.items()}
    chunks = []
    chunks_bio = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq_bio):
        if tok == default and chunk_start is not None:
            chunk = (chunk_start, i)
            chunks_bio.append(chunk)
            if chunk_type != "O":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
            chunk_start = None

        elif tok == bgn:
            if chunk_start is not None:
                chunk = (chunk_start, i)
                chunks_bio.append(chunk)
                if chunk_type != "O":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                chunk_start = None
            chunk_start = i
        # elif tok == inner:
        #     if chunk_start is None:
        #         chunk_start = i

            # tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            # if chunk_start is None:
            #     if tok_chunk_class != "I":
            #         chunk_start = i
            #     else:
            #         pass
            # elif tok_chunk_type != chunk_type:
            #     chunk = (chunk_type, chunk_start, i)
            #     chunks.append(chunk)
            #     if tok_chunk_class != "I":
            #         chunk_type, chunk_start = tok_chunk_type, i
            #     else:
            #         chunk_type, chunk_start = None, None

        else:
            pass
        chunk_type = idx_to_type[seq_type[i].item()]

    if chunk_start is not None:
        chunk = (chunk_start, len(seq_bio))
        chunks_bio.append(chunk)
        if chunk_type != "O":
            chunk = (chunk_type, chunk_start, len(seq_bio))
            chunks.append(chunk)
    
    # p_lst = []
    # for t,b in zip(seq_type, seq_bio):
    #     tp = idx_to_type[t.item()]
    #     sp = idx_to_tag[b.item()]
    #     if sp == "O":
    #         p_lst.append(sp)
    #     else:
    #         if tp != "O":
    #             p_lst.append(sp+"-"+tp)
    #         else:
    #             p_lst.append("O")


    return chunks, chunks_bio

def get_chunks_token(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # if tok == default and chunk_type is not None:
        #     chunk = (chunk_type, chunk_start, i)
        #     chunks.append(chunk)
        #     chunk_type, chunk_start = None, None
        if tok != default:
            chunk = (idx_to_tag[tok], i)      
            chunks.append(chunk)
        else:
            pass

    return chunks

def get_target_preds(threshold, x):
    top_prob, top_label = torch.topk(F.softmax(x, dim=1), k=1)
    top_label = top_label.squeeze().t()
    top_prob = top_prob.squeeze().t()
    top_mean, top_std = top_prob.mean(), top_prob.std()
    threshold_m = top_mean - threshold * top_std
    return top_label, top_prob, threshold_m

def soft_label(x):
    # x: batch_size*seq_len, num_labels
    x = F.softmax(x, dim=-1)
    weight = x**2 / torch.sum(x, dim=0) 
    target_distribution = (weight.t() / torch.sum(weight, dim=-1)).t()

    return target_distribution

if __name__ == '__main__':
    save(args)