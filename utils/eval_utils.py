import numpy as np
import logging
import os
import json
import tensorflow as tf
from label_mappings import *
from collections import defaultdict


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


class Result(object):
    def __init__(self, tokenizer):
        self.result = []
        self.tokenizer = tokenizer

    def get_result(self, model_output, tokenized_texts, origin_texts, language, tagging_schema="BIOES", result_type="end_to_end"):
        if result_type == "end_to_end":
            return self.get_result_end_to_end(model_output, tokenized_texts, origin_texts, language, tagging_schema=tagging_schema)
        else:
            return self.get_result_pipeline(model_output, tokenized_texts, origin_texts, language)

    def get_result_end_to_end(self, model_output, tokenized_texts, origin_texts, language, tagging_schema="BIOES"):
        if language == "en":
            label_category_mapping = LABEL_CATEGORY_MAPPING
            label_sentiment_mapping = LABEL_SENTIMENT_MAPPING
        else:
            label_category_mapping = LABEL_CATEGORY_MAPPING_CHINESE
            label_sentiment_mapping = LABEL_SENTIMENT_MAPPING_CHINESE
        # (batch_decoded_sequence, _), batch_cls_logits = model_output
        batch_decoded_sequence = model_output['decoded_sequence']
        batch_cls_logits = model_output['output_cls_states']
        batch_output_prob = softmax(model_output['output_logits'].numpy(), axis=-1)
        batch_decoded_sequence = batch_decoded_sequence.numpy()
        batch_cls_logits = batch_cls_logits.numpy()
        batch_cls_predict = (batch_cls_logits > 0).astype(np.int32)

        num_aspects = len(label_category_mapping)
        num_sentiments = len(label_sentiment_mapping)
        num_asp_senti_pairs = num_aspects * num_sentiments
        batch_size = len(origin_texts)
        for i in range(batch_size):
            # result = []
            result = set()
            decoded_sequence = batch_decoded_sequence[i]
            output_prob = batch_output_prob[i]
            cls_predict = batch_cls_predict[i]
            decoded_dict = decode_ner_output(decoded_sequence) if tagging_schema == "BIOES" else decode_ner_BIO_output(decoded_sequence)
            offset_mapping = tokenized_texts['offset_mapping'][i]
            origin_text = origin_texts[i]
            for asp_senti_idx in range(num_asp_senti_pairs):
                target_lists = decoded_dict[asp_senti_idx]
                target_aspect = label_category_mapping[asp_senti_idx // num_sentiments]
                target_sentiment = label_sentiment_mapping[asp_senti_idx % num_sentiments]
                if not target_lists and cls_predict[asp_senti_idx] == 1:  # implicit target
                    result.add((Triplet("NULL", target_aspect, target_sentiment, -1)))
                    continue
                for start, end in target_lists:
                    if offset_mapping[start][0] == offset_mapping[end - 1][1] == 0:  # [PAD] or [CLS] or [SEP]
                        continue

                    # target_text = self.tokenizer.convert_tokens_to_string(tokenized_text[start: end])
                    target_tag = decoded_sequence[asp_senti_idx][start: end]
                    target_prob = output_prob[asp_senti_idx][list(range(start, end)), target_tag]
                    avg_target_prob = np.mean(target_prob)
                    target_text = origin_text[offset_mapping[start][0]: offset_mapping[end - 1][1]]
                    result.add((Triplet(target_text, target_aspect, target_sentiment, avg_target_prob.item())))
            self.result.append(list(result))
        return self.result

    def get_result_pipeline(self, target_aspect_sentiments, tokenized_texts, origin_texts, language="en"):
        # 先获取标注结果
        batch_size = len(origin_texts)
        if "output" not in target_aspect_sentiments:
            return []
        target_aspect_sentiments = target_aspect_sentiments["output"].numpy()
        for i in range(batch_size):
            offset_mapping = tokenized_texts['offset_mapping'][i]
            origin_text = origin_texts[i]
            target_aspect_sentiment = target_aspect_sentiments[target_aspect_sentiments[:, -1] == i]
            self.result.append(
                self.parse_single_texts(target_aspect_sentiment, offset_mapping, origin_text))
        return self.result

    def parse_single_texts(self, target_aspect_sentiment, offset_mapping, origin_text, language="en"):
        tmp = defaultdict(list)
        result = []
        for triplet in target_aspect_sentiment:
            # target = self.tokenizer.convert_tokens_to_string(tokenized_texts[triplet[0]: triplet[1]].tolist())
            if triplet[0] == 0 and triplet[1] == 1:
                target = "[CLS]"
            else:
                target = origin_text[offset_mapping[triplet[0]][0]: offset_mapping[triplet[1] - 1][1]]
            aspect = LABEL_CATEGORY_MAPPING[triplet[2]]
            sentiment = LABEL_SENTIMENT_MAPPING[triplet[4]]
            tmp[target].append([aspect, sentiment])

        explict_targets_aspects = set()
        for target, aspect_senti_pairs in tmp.items():
            if target != "[CLS]":
                explict_targets_aspects = explict_targets_aspects.union([item[0] for item in aspect_senti_pairs])
                for aspect, sentiment in aspect_senti_pairs:
                    result.append(Triplet(target, aspect, sentiment))

        if '[CLS]' in tmp:
            for cls_aspect_senti_pairs in tmp["[CLS]"]:
                if cls_aspect_senti_pairs[0] not in explict_targets_aspects:
                    result.append(Triplet("NULL", cls_aspect_senti_pairs[0], cls_aspect_senti_pairs[1]))
        return result


def decode_ner_output(decoded_sequence):
    """

    :param decoded_sequence: (num_aspect_senti_pairs, seq_len)
    :param tokenized_text: (seq, len)
    :return: dict(asp_senti_idx: [start, end] list)
    """
    decoded_result = {}
    for asp_senti_idx, tag_sequence in enumerate(decoded_sequence):
        target_lists = []
        have_prefix = False
        entity_start = 0
        tag_idx = 0
        for tag_idx, tag in enumerate(tag_sequence):
            if tag == NER_LABEL_MAPPING["S"]:
                if have_prefix:
                    target_lists.append((entity_start, tag_idx + 1))
                    have_prefix = False
                else:
                    target_lists.append((tag_idx, tag_idx + 1))

            elif tag == NER_LABEL_MAPPING["B"]:
                if have_prefix:
                    target_lists.append((entity_start, tag_idx))
                have_prefix = True
                entity_start = tag_idx
            elif tag == NER_LABEL_MAPPING["I"]:
                #  Model predict "IIIE" instead of "BIIE"
                if not have_prefix:
                    entity_start = tag_idx
                    have_prefix = True

            elif tag == NER_LABEL_MAPPING["O"]:
                if have_prefix:
                    target_lists.append((entity_start, tag_idx))
                    have_prefix = False
            else:  # E
                if have_prefix:
                    target_lists.append((entity_start, tag_idx + 1))
                    have_prefix = False

        if have_prefix:
            target_lists.append((entity_start, tag_idx + 1))
        decoded_result[asp_senti_idx] = target_lists
    return decoded_result

def decode_ner_BIO_output(decoded_sequence):
    """

    :param decoded_sequence: (num_aspect_senti_pairs, seq_len)
    :param tokenized_text: (seq, len)
    :return: dict(asp_senti_idx: [start, end] list)
    """
    decoded_result = {}
    for asp_senti_idx, tag_sequence in enumerate(decoded_sequence):
        target_lists = []
        have_prefix = False
        entity_start = 0
        tag_idx = 0
        for tag_idx, tag in enumerate(tag_sequence):
            if tag == NER_BIO_MAPPING["B"]:
                if have_prefix:
                    target_lists.append((entity_start, tag_idx))
                have_prefix = True
                entity_start = tag_idx
            elif tag == NER_BIO_MAPPING["I"]:
                #  Model predict "III" instead of "BII"
                if not have_prefix:
                    entity_start = tag_idx
                    have_prefix = True
            else: # tag == NER_BIO_MAPPING["O"]
                if have_prefix:
                    target_lists.append((entity_start, tag_idx))
                    have_prefix = False

        if have_prefix:
            target_lists.append((entity_start, tag_idx + 1))
        decoded_result[asp_senti_idx] = target_lists
    return decoded_result


class Triplet(object):
    def __init__(self, target=None, aspect=None, sentiment=None, prob=None):
        self.__target = target
        self.__aspect = aspect
        self.__sentiment = sentiment
        self.__prob = prob

    def __eq__(self, other):
        return self.target == other.target and self.aspect == other.aspect and self.sentiment == other.sentiment

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.target, self.aspect, self.sentiment))

    @property
    def target(self):
        return self.__target

    @property
    def aspect(self):
        return self.__aspect

    @property
    def sentiment(self):
        return self.__sentiment

    @property
    def prob(self):
        return self.__prob

    @target.setter
    def target(self, target):
        self.__target = target

    @aspect.setter
    def aspect(self, aspect):
        self.__aspect = aspect

    @sentiment.setter
    def sentiment(self, sentiment):
        self.__sentiment = sentiment

    @prob.setter
    def prob(self, prob):
        self.__prob = prob

    def __str__(self) -> str:
        return json.dumps(self.json(), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()
    
    def json(self):
        return {
            'target': self.target,
            'aspect': self.aspect,
            'polarity': self.sentiment,
            'prob': self.prob
        }


def compute_f1(predicts, labels):
    tp = 0
    pos = 0
    true = 0
    for p, l in zip(predicts, labels):
        true += len(l)
        pos += len(p)
        tp += len(p & l)
    precision = tp / pos if pos != 0 else 0
    recall = tp / true if true != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1


def quick_test(test_data, test_tokenizer, trainer, output_attentions=False):
    data = test_tokenizer.tokenize(
        test_data
    )

    if trainer.model_type == 'single_tower':
        inputs = [data['input_ids'], data['token_type_ids'], data['position_ids'], data['attention_mask']]
        out = trainer.model(inputs, phase="test", output_attentions=output_attentions)
    elif trainer.model_type == 'double_tower':
        inputs = [data['input_ids'], data['token_type_ids'], data['position_ids'], data['attention_mask']]
        aspect_inputs = [data['aspect_input_ids'], data['aspect_token_type_ids'], data['aspect_attention_mask']]
        out = trainer.model(inputs, aspect_inputs, phase="test", output_attentions=output_attentions)
    else:
        inputs = [data['input_ids'], data['token_type_ids'], data['attention_mask']]
        aspect_inputs = [data['aspect_input_ids'], data['aspect_token_type_ids'], data['aspect_attention_mask']]
        signatures = {
            "text_inputs": [tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                            tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)],
            "aspect_inputs": [tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                              tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                              tf.TensorSpec(shape=(None, None), dtype=tf.int32)]
        }

        model_fun = trainer.model.call.get_concrete_function(
            text_inputs=signatures['text_inputs'], aspect_inputs=signatures['aspect_inputs'], label_inputs=None,
            phase="test", output_attentions=output_attentions)
        out = model_fun(inputs, aspect_inputs, phase="test", output_attentions=output_attentions)

        # out = trainer.model(inputs, aspect_inputs, phase="test", output_attentions=output_attentions)

    res = Result(test_tokenizer.tokenizer)
    return (res.get_result(out, data, test_data, language=trainer.language, result_type=trainer.model_type),
            out)
# if not output_attentions else (
#        res.get_result(out, data, test_data), out)

def save_predict(contexts, predicts, labels, file_path):
    res_list = []
    for t, p, l in zip(contexts, predicts, labels):
        res = {
            "context": t,
            "predict": [each.json() for each in p],
            "label":[each.json() for each in l]
        }
        res_list.append(res)
    json.dump(res_list, open(file_path, "w"))

