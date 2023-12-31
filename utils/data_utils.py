import numpy as np
import logging
import os
import json
from label_mappings import *
from collections import defaultdict


def prepare_logger(logger, level=logging.INFO, save_to_file=None, stream=True):
    """logger wraper"""
    formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s')
    if stream:
        console_hdl = logging.StreamHandler()
        console_hdl.setFormatter(formatter)
        logger.addHandler(console_hdl)
    if save_to_file is not None:  # and not os.path.exists(save_to_file):
        file_hdl = logging.FileHandler(save_to_file, mode="w")
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    logger.setLevel(level)
    logger.propagate = False


def infer_tokenized_label_old(text, tokenized_text, start, end):
    if start == end == 0:
        return -1, -1
    new_start = -1
    new_end = -1
    origin_pointer = 0
    new_target = ""
    for idx, token in enumerate(tokenized_text[1: -1], 1):
        if origin_pointer == start:
            new_start = idx

        if token.startswith("##"):
            sub_token = token.lstrip("##")
            assert text[origin_pointer: origin_pointer + len(sub_token)] == sub_token
            origin_pointer += len(sub_token)
            if new_start != -1 and new_end == -1:
                new_target += sub_token
        else:
            assert text[origin_pointer: origin_pointer + len(token)] == token
            origin_pointer += len(token)
            if new_start != -1 and new_end == -1:
                new_target += token

        if origin_pointer == end:
            new_end = idx

        while origin_pointer < len(text) and text[origin_pointer] == " ":
            origin_pointer += 1
            if new_start != -1 and new_end == -1:
                new_target += " "

    if new_start == -1 or new_end == -1:
        raise ValueError('Sentence {} / {} Matching failed!'.format(text, tokenized_text))
    elif new_target != text[start: end]:
        raise ValueError('{} is not equal to {}'.format(new_target, text[start: end]))
    else:
        return new_start, new_end


def infer_tokenized_label(start, end, offsets_mapping):
    if start == end == 0:
        return 0, 1

    new_start = -1
    new_end = -1
    for idx, offset in enumerate(offsets_mapping[1: -1], 1):
        if new_start == -1:
            if offset[0] == start:
                new_start = idx

        if new_start != -1 and new_end == -1:
            if offset[1] == end:
                new_end = idx + 1
    if new_start == -1 or new_end == -1:
        raise ValueError('Sentence {} / {} Matching failed!')
    else:
        return new_start, new_end

def format_tokenized_label(start, end):
    if start == end == -1:
        return 0, 1
    else:
        return start + 1, end + 1

def convert_batch_label_to_batch_tokenized_label(offsets_mapping, triplets, max_len):
    # tokens_id = tokenizer(text, return_offsets_mapping=True)
    # offsets_mapping = tokenized_inputs['offset_mapping']
    # tokenized_text = tokenizer.convert_ids_to_tokens(tokens_id['input_ids'])
    ner_label = [NER_LABEL_MAPPING['O']] * len(offsets_mapping)
    classify_label = []
    sentiment_label = []
    history_target = []
    if not triplets:
        classify_label.append([0, 1] + [0] * NUM_CATEGORIES)
        sentiment_label.append([0, 1, 0, 1, -1])
    else:
        target_aspect_dict = {(0, 1): [0] * NUM_CATEGORIES}
        for triplet in triplets:
            category = triplet['category']
            polarity = triplet['polarity']
            start, end = int(triplet['from']), int(triplet['to'])
            new_start, new_end = infer_tokenized_label(start, end, offsets_mapping)

            if (new_start, new_end) not in target_aspect_dict:
                target_aspect_dict[(new_start, new_end)] = [0] * NUM_CATEGORIES
            # create label
            if new_start == 0 and new_end == 1:
                target_aspect_dict[(0, 1)][CATEGORY_LABEL_MAPPING[category]] = 1
                # classify_label.append([-1, -1, CATEGORY_LABEL_MAPPING[category]])
                sentiment_label.append(
                    [0, 1]
                    + [CATEGORY_LABEL_MAPPING[category], CATEGORY_LABEL_MAPPING[category] + 1]
                    + [SENTIMENT_LABEL_MAPPING[polarity]]
                )
            else:
                if new_start + 1 == new_end:
                    assert ner_label[new_start] == NER_LABEL_MAPPING['O'] or (start, end) in history_target, triplets
                    ner_label[new_start] = NER_LABEL_MAPPING['S']
                elif new_start + 2 == new_end:
                    assert (ner_label[new_start] == NER_LABEL_MAPPING['O']
                            and ner_label[new_end - 1] == NER_LABEL_MAPPING['O']) or (start, end) in history_target
                    ner_label[new_start] = NER_LABEL_MAPPING['B']
                    ner_label[new_end - 1] = NER_LABEL_MAPPING['E']
                else:
                    assert (ner_label[new_start: new_end] == [NER_LABEL_MAPPING['O']] * (new_end - new_start)) \
                           or (start, end) in history_target
                    ner_label[new_start] = NER_LABEL_MAPPING['B']
                    ner_label[new_end - 1] = NER_LABEL_MAPPING['E']
                    ner_label[new_start + 1: new_end - 1] = [NER_LABEL_MAPPING['I']] * (new_end - new_start - 2)
                # classify_label.append([new_start, new_end, CATEGORY_LABEL_MAPPING[category]])
                target_aspect_dict[(new_start, new_end)][CATEGORY_LABEL_MAPPING[category]] = 1
                target_aspect_dict[(0, 1)][CATEGORY_LABEL_MAPPING[category]] = 1
                # sentiment_label.append([new_start, new_end, CATEGORY_LABEL_MAPPING[category],
                #                         SENTIMENT_LABEL_MAPPING[polarity]])
                sentiment_label.append(
                    [new_start, new_end]
                    + [CATEGORY_LABEL_MAPPING[category], CATEGORY_LABEL_MAPPING[category] + 1]
                    + [SENTIMENT_LABEL_MAPPING[polarity]]
                )
            history_target.append((start, end))
        # classify label
        for (start, end), l in target_aspect_dict.items():
            classify_label.append([start, end] + l)

    # ner_label = ner_label + [-1] * (max_len - len(ner_label))

    final_result = [ner_label, classify_label, sentiment_label]

    return final_result


def convert_batch_label_to_batch_tokenized_label_end_to_end(offsets_mapping, aspect_sentiment_mapping, triplets):
    sentiment_label_mapping = {s: idx for idx, s in enumerate(aspect_sentiment_mapping['sentiments'])}
    category_label_mapping = aspect_sentiment_mapping['category2index']
    num_aspects = len(category_label_mapping)
    num_sentiments = len(sentiment_label_mapping)
    ner_label = [[NER_LABEL_MAPPING['O']] * len(offsets_mapping) for _ in range(num_aspects * num_sentiments)]
    cls_label = [0] * (num_aspects * num_sentiments)
    if triplets:
        for triplet in triplets:
            aspect_idx = category_label_mapping[triplet['category']]
            senti_idx = sentiment_label_mapping[triplet['polarity']]
            aspect_senti_idx = aspect_idx * num_sentiments + senti_idx
            start, end = int(triplet['from']), int(triplet['to'])
            new_start, new_end = infer_tokenized_label(start, end, offsets_mapping)

            # create label
            cls_label[aspect_senti_idx] = 1
            if new_start == 0 and new_end == 1:
                continue
            else:
                if new_start + 1 == new_end:
                    assert ner_label[aspect_senti_idx][new_start] == NER_LABEL_MAPPING['O']
                    ner_label[aspect_senti_idx][new_start] = NER_LABEL_MAPPING['S']
                elif new_start + 2 == new_end:
                    assert (ner_label[aspect_senti_idx][new_start] == NER_LABEL_MAPPING['O']
                            and ner_label[aspect_senti_idx][new_end - 1] == NER_LABEL_MAPPING['O']), triplets
                    ner_label[aspect_senti_idx][new_start] = NER_LABEL_MAPPING['B']
                    ner_label[aspect_senti_idx][new_end - 1] = NER_LABEL_MAPPING['E']
                else:
                    assert (ner_label[aspect_senti_idx][new_start: new_end] == [NER_LABEL_MAPPING['O']] * (
                            new_end - new_start))
                    ner_label[aspect_senti_idx][new_start] = NER_LABEL_MAPPING['B']
                    ner_label[aspect_senti_idx][new_end - 1] = NER_LABEL_MAPPING['E']
                    ner_label[aspect_senti_idx][new_start + 1: new_end - 1] = [NER_LABEL_MAPPING['I']] * (
                            new_end - new_start - 2)

    final_result = [ner_label, cls_label]
    return final_result



def convert_batch_label_to_batch_tokenized_label_end_to_end_BIO(offsets_mapping, triplets, aspect_sentiment_mapping, tokenized_label=False, origin_text=None):
    sentiment_label_mapping = {s: idx for idx, s in enumerate(aspect_sentiment_mapping['sentiments'])}    
    category_label_mapping = aspect_sentiment_mapping['category2index']
    num_aspects = len(category_label_mapping)
    num_sentiments = len(sentiment_label_mapping)
    ner_label = [[NER_BIO_MAPPING['O']] * len(offsets_mapping) for _ in range(num_aspects * num_sentiments)]
    cls_label = [0] * (num_aspects * num_sentiments)
    if triplets:
        for triplet in triplets:
            aspect_idx = category_label_mapping[triplet['category']]
            senti_idx = sentiment_label_mapping[triplet['polarity']]
            aspect_senti_idx = aspect_idx * num_sentiments + senti_idx
            start, end = int(triplet['from']), int(triplet['to'])
            if tokenized_label:
                new_start, new_end = format_tokenized_label(start, end)
            else:
                new_start, new_end = infer_tokenized_label(start, end, offsets_mapping) 

            # create label
            cls_label[aspect_senti_idx] = 1
            if new_start == 0 and new_end == 1:
                continue
            else:
                if origin_text:
                    # assert origin_text[offsets_mapping[new_start][0]: offsets_mapping[new_end - 1][1]] == triplet['target']
                    if origin_text[offsets_mapping[new_start][0]: offsets_mapping[new_end - 1][1]] != triplet['target']:
                        print(origin_text)
                if new_start + 1 == new_end:
                    assert ner_label[aspect_senti_idx][new_start] == NER_BIO_MAPPING['O']
                    ner_label[aspect_senti_idx][new_start] = NER_BIO_MAPPING['B']
                else:
                    try:
                        assert (ner_label[aspect_senti_idx][new_start: new_end] == [NER_BIO_MAPPING['O']] * (
                            new_end - new_start))
                    except:
                        print(origin_text)
                        continue
                    ner_label[aspect_senti_idx][new_start] = NER_BIO_MAPPING['B']
                    ner_label[aspect_senti_idx][new_start + 1: new_end] = [NER_BIO_MAPPING['I']] * (
                            new_end - new_start - 1)

    final_result = [ner_label, cls_label]
    return final_result