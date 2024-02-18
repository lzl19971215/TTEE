from collections import namedtuple
import json
import re
import numpy as np
import tensorflow as tf
import pandas as pd
import random
from xml.etree import ElementTree
from utils.data_utils import convert_batch_label_to_batch_tokenized_label, \
    convert_batch_label_to_batch_tokenized_label_end_to_end, convert_batch_label_to_batch_tokenized_label_end_to_end_BIO
from label_mappings import RES1516_LABEL_MAPPING, ACOS_LAPTOP_LABEL_MAPPING
from itertools import chain


class BaseSemEvalDataSet(object):

    def __init__(self, file_path, tokenizer, sentence_b, mask_sb=False, model_type="single_tower", tagging_schema="BIOES", drop_null=False, is_label_after_tokenized=False, neg_sample=-1, data_sample_ratio=-1):
        self.tokenizer = tokenizer
        tokenizer_type = type(tokenizer).__name__
        self.is_strict = re.search('deberta', tokenizer_type.lower()) is None
        self.sentence_b = sentence_b
        # self.tree = ElementTree.parse(file_path)
        # self.root = self.tree.getroot()
        self.file_path = file_path
        self.drop_null = drop_null
        self.tagging_schema = tagging_schema
        self.string_sentences = self.xml2list()
        if self.drop_null:
            self.string_sentences = [each for each in self.string_sentences if len(json.loads(each[1]))]
        if data_sample_ratio != -1:
            self.string_sentences = random.sample(self.string_sentences, int(data_sample_ratio * len(self.string_sentences)))
        self.mask_sb = mask_sb
        self.model_type = model_type
        self.is_label_after_tokenized = is_label_after_tokenized
        self.asp_senti_to_idx, self.aspect_texts, self.sentiment_texts, self.sb_pos_ids = None, None, None, None
        if self.model_type == "end_to_end":
            self.asp_senti_to_idx, self.aspect_texts, self.sentiment_texts = self._create_aspect_sentiment_pairs()
        elif self.model_type == "single_tower":
            self.sb_pos_ids = self._infer_sb_pos_ids()
        self.language = None
        self.neg_sample = neg_sample

    def __getitem__(self, item):
        return self.string_sentences[item]

    def _create_aspect_sentiment_pairs(self):
        aspect_texts = self.sentence_b['texts']
        num_aspects = len(aspect_texts)
        sentiment_texts = self.sentence_b['sentiments']
        num_sentiments = len(sentiment_texts)
        aspect_texts = [[aspect] * num_sentiments for aspect in aspect_texts]
        aspect_texts = list(chain(*aspect_texts))
        sentiment_texts = sentiment_texts * num_aspects
        assert len(aspect_texts) == len(sentiment_texts)
        asp_senti_to_idx = {idx: (aspect, senti) for idx, (aspect, senti) in
                            enumerate(zip(aspect_texts, sentiment_texts))}
        return asp_senti_to_idx, aspect_texts, sentiment_texts

    def _infer_sb_pos_ids(self):
        sb_pos_ids = []
        for i, (start, end) in self.sentence_b['aspect_term_mapping'].items():
            sb_pos_ids.extend(list(range(end - start)))
        return sb_pos_ids

    def __len__(self):
        return len(self.string_sentences)

    def generate_string_sample(self):
        for ss in self.string_sentences:
            yield ss

    def generate_attention_mask(self, attention_mask, sb_len, mask_sentence_b=False):
        if not mask_sentence_b:
            return attention_mask + sb_len * [1]

        sa_len = len(attention_mask)
        sa_mask = np.array(attention_mask + sb_len * [0])[np.newaxis, :]
        sa_mask = np.repeat(sa_mask, sa_len, axis=0)  # (sa_len, sa_len + sb_len)

        left_mask = np.repeat(np.array(attention_mask)[np.newaxis, :], sb_len, axis=0)
        right_mask = np.zeros((sb_len, sb_len), dtype=int)
        for i, (start, end) in self.sentence_b['aspect_term_mapping'].items():
            index_grid = np.ix_(list(range(start, end)), list(range(start, end)))
            right_mask[index_grid] = np.ones_like(right_mask[index_grid], dtype=int)
        sb_mask = np.hstack([left_mask, right_mask])
        mask = np.vstack([sa_mask, sb_mask])
        mask[-1][sa_len:] = np.ones(sb_len)
        return mask

    def map_batch_string_to_tensor(self, text_tensor, triplet_tensor):
        """

        :param text_tensor:
        :param triplet_tensor:
        :return: [input_ids, token_type_ids, position_ids, attention_mask, ner_labels, classify_labels,
                sentiment_labels, batch_sentiment_index, batch_clf_index]
        """
        batch_size = text_tensor.shape[0]
        origin_texts = [text.decode('utf-8') for text in text_tensor.numpy()]
        tokenized_texts = self.tokenizer(
            origin_texts,
            padding='longest',
            return_offsets_mapping=True
        )
        tokenized_sentence_b = self.tokenizer(
            self.sentence_b['text'],
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False
        )
        s_b_len = len(tokenized_sentence_b['input_ids']) + 1
        max_len = len(tokenized_texts['input_ids'][0])
        input_ids = []
        token_type_ids = []
        position_ids = []
        attention_mask = []
        ner_labels = []
        classify_labels = []
        sentiment_labels = []
        batch_senti_index = []
        batch_clf_index = []
        for i in range(batch_size):
            triplet = json.loads(triplet_tensor[i].numpy())
            num_triplet = len(triplet) if triplet else 1
            offsets_mapping = tokenized_texts['offset_mapping'][i]
            ner_label, classify_label, sentiment_label = convert_batch_label_to_batch_tokenized_label(
                offsets_mapping=offsets_mapping,
                triplets=triplet,
                max_len=max_len
            )

            one_input_ids = tokenized_texts['input_ids'][i] + tokenized_sentence_b['input_ids'] + [102]
            input_ids.append([one_input_ids])

            one_token_type_ids = tokenized_texts['token_type_ids'][i] + [1] * s_b_len
            token_type_ids.append([one_token_type_ids])

            one_position_ids = [i for i in range(max_len)] + [i for i in self.sb_pos_ids] + \
                               [max_len]
            position_ids.append([one_position_ids])

            if not self.mask_sb:
                one_attention_mask = [tokenized_texts['attention_mask'][i] + [1] * s_b_len] * (s_b_len + max_len)
            else:
                one_attention_mask = self.generate_attention_mask(tokenized_texts['attention_mask'][i], s_b_len,
                                                                  mask_sentence_b=True)
            attention_mask.append([one_attention_mask])

            ner_label = ner_label + [-1] * s_b_len
            ner_labels.append([ner_label])

            classify_labels.append(classify_label)
            sentiment_labels.append(sentiment_label)
            batch_senti_index.extend([i] * len(sentiment_label))

            batch_clf_index.extend([i] * len(classify_label))

        result = [np.concatenate(input_ids, axis=0), np.concatenate(token_type_ids, axis=0),
                  np.concatenate(position_ids, axis=0), np.concatenate(attention_mask, axis=0),
                  np.concatenate(ner_labels, axis=0), np.concatenate(classify_labels, axis=0),
                  np.concatenate(sentiment_labels, axis=0), np.array(batch_senti_index), np.array(batch_clf_index)]

        result[-5] = tf.where(result[0] == 0, tf.fill(result[0].shape, -1), result[-5])
        return result

    def map_batch_string_to_tensor_double_tower(self, text_tensor, triplet_tensor):
        batch_size = text_tensor.shape[0]
        origin_texts = [text.decode('utf-8') for text in text_tensor.numpy()]
        tokenized_texts = self.tokenizer(
            origin_texts,
            padding='longest',
            return_offsets_mapping=True
        )
        aspect_texts = self.tokenizer(
            RES1516_LABEL_MAPPING['texts'],
            padding='longest',
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )
        max_len = len(tokenized_texts['input_ids'][0])
        # input_ids = []
        # token_type_ids = []
        position_ids = []
        # attention_mask = []
        ner_labels = []
        classify_labels = []
        sentiment_labels = []
        batch_senti_index = []
        batch_clf_index = []
        for i in range(batch_size):
            triplet = json.loads(triplet_tensor[i].numpy())
            num_triplet = len(triplet) if triplet else 1
            offsets_mapping = tokenized_texts['offset_mapping'][i]
            ner_label, classify_label, sentiment_label = convert_batch_label_to_batch_tokenized_label(
                offsets_mapping=offsets_mapping,
                triplets=triplet,
                max_len=max_len
            )

            one_position_ids = [i for i in range(max_len)]
            position_ids.append([one_position_ids])

            ner_labels.append([ner_label])

            classify_labels.append(classify_label)
            sentiment_labels.append(sentiment_label)
            batch_senti_index.extend([i] * len(sentiment_label))

            batch_clf_index.extend([i] * len(classify_label))

        attention_mask = np.array(tokenized_texts['attention_mask'])
        attention_mask = np.repeat(attention_mask[:, np.newaxis, :], attention_mask.shape[1], axis=1)
        result = [
            np.array(tokenized_texts['input_ids']),
            np.array(tokenized_texts['token_type_ids']),
            np.concatenate(position_ids, axis=0),
            attention_mask,
            np.concatenate(ner_labels, axis=0),
            np.concatenate(classify_labels, axis=0),
            np.concatenate(sentiment_labels, axis=0),
            np.array(batch_senti_index),
            np.array(batch_clf_index)
        ]
        aspect_result = [
            np.array(aspect_texts['input_ids']),
            np.array(aspect_texts['token_type_ids']),
            np.array(aspect_texts['attention_mask'])
        ]

        result[-5] = tf.where(result[0] == 0, -1, result[-5])
        return result + aspect_result

    # def negative_sampling(self, labels, n_sample=32):
    #     num_rows, _ = labels.shape
    #     sampled_indices = []
        
    #     for i in range(num_rows):
    #         positive_indices = np.where(labels[i] == 1)[0]
    #         negative_indices = np.where(labels[i] == 0)[0]
    #         assert len(positive_indices) < n_sample

    #         chosen_indices = positive_indices
    #         num_negatives_needed = n_sample - len(chosen_indices)
    #         chosen_negative_indices = np.random.choice(negative_indices, size=num_negatives_needed, replace=False)
    #         chosen_indices = np.concatenate((chosen_indices, chosen_negative_indices))
            
    #         # 打乱选择的索引，使正负样本混合
    #         sampled_indices.append(chosen_indices)
        
    #     batch_index = np.repeat(np.arange(num_rows)[:, None], n_sample, axis=1)
    #     return batch_index, np.array(sampled_indices)
    def negative_sampling(self, labels, n_sample=36):
        _, n_cols = labels.shape
        positive_indices = np.unique(np.nonzero(labels)[1])
        negative_indices = np.array([i for i in range(n_cols) if i not in positive_indices])
        num_negatives_needed = n_sample - len(positive_indices) 
        if num_negatives_needed <= 0:
            return positive_indices
        chosen_negative_indices = np.random.choice(negative_indices, size=num_negatives_needed, replace=False)
        chosen_indices = np.concatenate((positive_indices, chosen_negative_indices))
        return chosen_indices    

    def map_batch_string_to_tensor_end_to_end(self, text_tensor, triplet_tensor):
        batch_size = text_tensor.shape[0]
        origin_texts = [text.decode('utf-8') for text in text_tensor.numpy()]
        tokenized_texts = self.tokenizer(
            origin_texts,
            padding='longest',
            return_offsets_mapping=True
        )

        aspect_senti_inputs = self.tokenizer(
            self.aspect_texts,
            self.sentiment_texts,
            padding='longest',
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )
        max_len = len(tokenized_texts['input_ids'][0])
        ner_labels = []
        cls_labels = []
        raw_label_to_tokenized_label_func = convert_batch_label_to_batch_tokenized_label_end_to_end if self.tagging_schema == "BIOES" else convert_batch_label_to_batch_tokenized_label_end_to_end_BIO 
        for i in range(batch_size):
            triplet = json.loads(triplet_tensor[i].numpy())
            offsets_mapping = tokenized_texts['offset_mapping'][i]
            ner_label, cls_label = raw_label_to_tokenized_label_func(
                offsets_mapping=offsets_mapping,
                triplets=triplet,
                aspect_sentiment_mapping=self.sentence_b,
                tokenized_label=self.is_label_after_tokenized,
                origin_text=origin_texts[i],
                strict=self.is_strict
            )

            # one_position_ids = [i for i in range(max_len)]
            # position_ids.append([one_position_ids])

            ner_labels.append([ner_label])
            cls_labels.append([cls_label])

        attention_mask = np.array(tokenized_texts['attention_mask'])
        all_ner_labels = np.concatenate(ner_labels, axis=0)
        all_cls_labels = np.concatenate(cls_labels, axis=0)
        aspect_senti_input_ids = np.array(aspect_senti_inputs['input_ids'])
        aspect_senti_token_type_ids = np.array(aspect_senti_inputs['token_type_ids'])
        aspect_senti_attention_mask = np.array(aspect_senti_inputs['attention_mask'])

        if self.neg_sample != -1:
            asp_senti_idx = self.negative_sampling(all_cls_labels, n_sample=self.neg_sample)
            all_ner_labels = all_ner_labels[:, asp_senti_idx]
            all_cls_labels = all_cls_labels[:, asp_senti_idx]
            aspect_senti_input_ids = aspect_senti_input_ids[asp_senti_idx]
            aspect_senti_token_type_ids = aspect_senti_token_type_ids[asp_senti_idx]
            aspect_senti_attention_mask = aspect_senti_attention_mask[asp_senti_idx]

            

        text_inputs_list = [
            np.array(tokenized_texts['input_ids']),
            np.array(tokenized_texts['token_type_ids']),
            attention_mask,
            all_ner_labels,
            all_cls_labels
        ]

        aspect_senti_inputs_list = [
            aspect_senti_input_ids,
            aspect_senti_token_type_ids,
            aspect_senti_attention_mask
        ]

        return text_inputs_list + aspect_senti_inputs_list

    def xml2list(self):
        raise NotImplementedError

    def wrap_map(self, text_tensor, triplet_tensor):
        if self.model_type == "single_tower":
            result = tf.py_function(
                self.map_batch_string_to_tensor,
                inp=[text_tensor, triplet_tensor],
                Tout=[tf.int32] * 9
            )
        elif self.model_type == "double_tower":
            result = tf.py_function(
                self.map_batch_string_to_tensor_double_tower,
                inp=[text_tensor, triplet_tensor],
                Tout=[tf.int32] * 12
            )
        else:
            result = tf.py_function(
                self.map_batch_string_to_tensor_end_to_end,
                inp=[text_tensor, triplet_tensor],
                Tout=[tf.int32] * 8
            )

        return result

    def augment(self, num_merge_list):
        all_merge_result = []
        for num_merge in num_merge_list:
            merge_result = self.merge_origin_sentences(num_merge)
            all_merge_result.extend(merge_result)
        self.string_sentences.extend(all_merge_result)

    def merge_origin_sentences(self, num_merge):
        merge_result = []
        for idx, (text, triplets) in enumerate(self.string_sentences):
            merge_sentences = [(text, triplets)]
            if idx < len(self.string_sentences) - num_merge:
                for i in range(1, num_merge + 1):
                    merge_sentences.append(self.string_sentences[idx + i])

            start = 0
            aspects = set()
            cur_merge_triplets = []
            cls_triplets = []
            aug_text = ""
            for i, (merge_text, merge_triplets) in enumerate(merge_sentences):
                if i == 0:
                    aug_text += merge_text
                    add = 0
                else:
                    aug_text += " " + merge_text
                    add = 1
                merge_triplets = json.loads(merge_triplets)
                for m_t in merge_triplets:
                    new_t = m_t.copy()
                    if new_t['target'] == "NULL":
                        cls_triplets.append(new_t)
                        continue
                    new_from = int(new_t['from']) + add + start
                    new_to = int(new_t['to']) + add + start
                    new_t['from'] = str(new_from)
                    new_t['to'] = str(new_to)
                    assert aug_text[new_from: new_to] == new_t['target'], "{} {}".format(
                        aug_text[new_from: new_to], new_t['target'])
                    cur_merge_triplets.append(new_t)
                    aspects.add(new_t['category'])
                start += len(merge_text) + add
            for cls_t in cls_triplets:
                if cls_t['category'] in aspects:
                    continue
                cur_merge_triplets.append(cls_t)
                aspects.add(cls_t['category'])

            merge_result.append((aug_text, json.dumps(cur_merge_triplets, ensure_ascii=False)))
        return merge_result


class EnglishDataset(BaseSemEvalDataSet):
    def __init__(self, file_path, tokenizer, sentence_b, mask_sb=False, tagging_schema="BIOES", model_type="end_to_end", drop_null=False, neg_sample=-1, data_sample_ratio=-1):
        super(EnglishDataset, self).__init__(file_path, tokenizer, sentence_b, mask_sb, model_type, tagging_schema, drop_null, False, neg_sample, data_sample_ratio)
        self.language = "en"

    def xml2list(self):
        # origin_result = []
        # tokenized_result = []
        tree = ElementTree.parse(self.file_path)
        root = tree.getroot()
        string_result = []
        for review in root:
            rid = review.attrib['rid']
            sentences = review[0]
            for sentence in sentences:
                sentence_dict = {}
                sid = sentence.attrib['id'].split(':')[-1]
                text = sentence[0].text
                sentence_dict['rid'] = rid
                sentence_dict['sid'] = sid
                sentence_dict['sentence_text'] = text
                sentence_dict['triplet'] = []

                if len(sentence) > 1:
                    opinions = sentence[1]
                    for opinion in opinions:
                        sentence_dict['triplet'].append(opinion.attrib)
                string_result.append((text, json.dumps(sentence_dict['triplet'])))

        return string_result

class ACOSDataset(BaseSemEvalDataSet):
    def __init__(self, file_path, tokenizer, sentence_b, mask_sb=False, tagging_schema="BIOES", model_type="end_to_end", drop_null=False, neg_sample=-1, data_sample_ratio=-1):
        self.Triplet = namedtuple('triplet', ['target', 'category', 'polarity', 'start', 'end'])
        super(ACOSDataset, self).__init__(file_path, tokenizer, sentence_b, mask_sb, model_type, tagging_schema, drop_null, is_label_after_tokenized=False, neg_sample=neg_sample, data_sample_ratio=data_sample_ratio)
        self.language = "en"

    
    def xml2list(self):
        string_result = []
        with open(self.file_path, encoding='utf-8') as f:
            for row in f:
                row = row.strip().split('\t')
                text = row[0]
                trips = row[1:]
                triplets = set()
                for trip in trips:
                    items = trip.split()
                    start, end = items[0].split(',')
                    target = "NULL" if start == end == '0' else text[int(start): int(end)]

                    category = items[1]
                    polarity = items[2]
                     
                    triplets.add(self.Triplet(target, category, polarity, start, end))
                triplets_list = []
                for tup in triplets:
                    triplets_list.append({
                        'target': tup.target,
                        'category': tup.category,
                        'polarity': tup.polarity,
                        'from': tup.start,
                        'to': tup.end
                    })
                string_result.append((text, json.dumps(triplets_list, ensure_ascii=False)))
        return string_result  

class ChineseDataset(BaseSemEvalDataSet):
    def __init__(self, file_path, tokenizer, sentence_b, mask_sb=False, tagging_schema="BIOES", model_type="end_to_end", drop_null=False, neg_sample=-1):
        super(ChineseDataset, self).__init__(file_path, tokenizer, sentence_b, mask_sb, model_type, tagging_schema, drop_null)
        self.language = "cn"

    def xml2list(self):
        df = pd.read_csv(self.file_path)
        string_result = []
        prev_text = ""
        triplets = []
        for idx, row in df.iterrows():
            text = row['text']
            target = row['target']
            category = row['category']
            polarity = row['polarity']
            start = row['from']
            end = row['to']

            if text != prev_text:
                triplets = []

            if not pd.isna(target):
                triplets.append({
                    'target': target, 'category': category, 'polarity': polarity, 'from': start, 'to': end
                })

            string_result.append((text, json.dumps(triplets, ensure_ascii=False)))
        return string_result


class TestTokenizer(object):
    def __init__(self, tokenizer, sentence_b, mask_sb=False, model_type="single_tower"):
        self.tokenizer = tokenizer
        self.sentence_b = sentence_b

        self.mask_sb = mask_sb
        self.model_type = model_type
        self.asp_senti_to_idx, self.aspect_texts, self.sentiment_texts, self.sb_pos_ids = None, None, None, None
        if self.model_type == "end_to_end":
            self.asp_senti_to_idx, self.aspect_texts, self.sentiment_texts = self._create_aspect_sentiment_pairs()
        elif self.model_type == "single_tower":
            self.sb_pos_ids = self._infer_sb_pos_ids()

    def _create_aspect_sentiment_pairs(self):
        aspect_texts = self.sentence_b['texts']
        num_aspects = len(aspect_texts)
        sentiment_texts = self.sentence_b['sentiments']
        num_sentiments = len(sentiment_texts)
        aspect_texts = [[aspect] * num_sentiments for aspect in aspect_texts]
        aspect_texts = list(chain(*aspect_texts))
        sentiment_texts = sentiment_texts * num_aspects
        assert len(aspect_texts) == len(sentiment_texts)
        asp_senti_to_idx = {idx: (aspect, senti) for idx, (aspect, senti) in
                            enumerate(zip(aspect_texts, sentiment_texts))}
        return asp_senti_to_idx, aspect_texts, sentiment_texts

    def tokenize(self, texts, aspect_texts=None, max_len=None):
        if self.model_type == "double_tower":
            result = self.tokenize_double_tower(texts)
        elif self.model_type == "single_tower":
            result = self.tokenize_single_tower(texts)
        else:
            result = self.tokenize_end_to_end(texts, aspect_texts=aspect_texts, max_len=max_len)
        return result     

    def tokenize_single_tower(self, texts):
        batch_size = len(texts)
        tokenized_texts = self.tokenizer(
            texts,
            padding='longest',
            return_offsets_mapping=True,
        )
        tokenized_sentence_b = self.tokenizer(
            self.sentence_b['text'],
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False
        )
        s_b_len = len(tokenized_sentence_b['input_ids']) + 1
        max_len = len(tokenized_texts['input_ids'][0])
        input_ids = []
        token_type_ids = []
        position_ids = []
        attention_mask = []
        for i in range(batch_size):
            one_input_ids = tokenized_texts['input_ids'][i] + tokenized_sentence_b['input_ids'] + [102]
            input_ids.append([one_input_ids])

            one_token_type_ids = tokenized_texts['token_type_ids'][i] + [1] * s_b_len
            token_type_ids.append([one_token_type_ids])

            one_position_ids = [i for i in range(max_len)] + [i for i in self.sb_pos_ids] + \
                               [max_len]
            position_ids.append([one_position_ids])

            if not self.mask_sb:
                one_attention_mask = [tokenized_texts['attention_mask'][i] + [1] * s_b_len] * (s_b_len + max_len)
            else:
                one_attention_mask = self.generate_attention_mask(tokenized_texts['attention_mask'][i], s_b_len,
                                                                  mask_sentence_b=True)
            attention_mask.append([one_attention_mask])

        result = {
            "input_ids": tf.concat(input_ids, axis=0),
            "token_type_ids": tf.concat(token_type_ids, axis=0),
            "position_ids": tf.concat(position_ids, axis=0),
            "attention_mask": tf.concat(attention_mask, axis=0),
            "offset_mapping": tokenized_texts["offset_mapping"]
        }
        return result

    def tokenize_double_tower(self, texts):
        batch_size = len(texts)
        tokenized_texts = self.tokenizer(
            texts,
            padding='longest',
            return_offsets_mapping=True
        )
        aspect_texts = self.tokenizer(
            self.sentence_b['text'],
            padding="longest",
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )

        max_len = len(tokenized_texts['input_ids'][0])
        position_ids = []
        for i in range(batch_size):
            one_position_ids = [i for i in range(max_len)]
            position_ids.append([one_position_ids])

        attention_mask = np.array(tokenized_texts['attention_mask'])
        attention_mask = np.repeat(attention_mask[:, np.newaxis, :], attention_mask.shape[1], axis=1)

        result = {
            "input_ids": tf.constant(tokenized_texts['input_ids']),
            "token_type_ids": tf.constant(tokenized_texts['token_type_ids']),
            "position_ids": tf.concat(position_ids, axis=0),
            "attention_mask": tf.constant(attention_mask),
            "aspect_input_ids": tf.constant(aspect_texts['input_ids']),
            "aspect_token_type_ids": tf.constant(aspect_texts['token_type_ids']),
            "aspect_attention_mask": tf.constant(aspect_texts['attention_mask'])
        }
        return result

    def tokenize_end_to_end(self, texts, aspect_texts=None, max_len=None):
        tokenized_texts = self.tokenizer(
            texts,
            padding='longest',
            return_offsets_mapping=True
        )
        if max_len is not None and len(tokenized_texts['input_ids'][0]) < max_len:
            tokenized_texts = self.tokenizer(
                texts,
                padding='max_length',
                return_offsets_mapping=True,
                max_length=max_len
            )

        if aspect_texts is None:
            aspect_senti_inputs = self.tokenizer(
                self.aspect_texts,
                self.sentiment_texts,
                padding='longest',
                add_special_tokens=True,
                return_token_type_ids=True,
                return_attention_mask=True
            )
        elif len(aspect_texts) == 1:
            aspect_senti_inputs = self.tokenizer(
                aspect_texts[0],
                padding='longest',
                add_special_tokens=True,
                return_token_type_ids=True,
                return_attention_mask=True
            )
        elif len(aspect_texts) == 2:
            aspect_senti_inputs = self.tokenizer(
                aspect_texts[0],
                aspect_texts[1],                
                padding='longest',
                add_special_tokens=True,
                return_token_type_ids=True,
                return_attention_mask=True
            ) 
        else:
            raise ValueError("wrong length of aspect_texts list")           

        attention_mask = np.array(tokenized_texts['attention_mask'])
        attention_mask = np.repeat(attention_mask[:, np.newaxis, :], attention_mask.shape[1], axis=1).astype(np.int32)

        result = {
            "input_ids": tf.constant(tokenized_texts['input_ids']),
            "token_type_ids": tf.constant(tokenized_texts['token_type_ids']),
            "attention_mask": tf.constant(attention_mask),
            "aspect_input_ids": tf.constant(aspect_senti_inputs['input_ids']),
            "aspect_token_type_ids": tf.constant(aspect_senti_inputs['token_type_ids']),
            "aspect_attention_mask": tf.constant(aspect_senti_inputs['attention_mask']),
            "offset_mapping": tokenized_texts['offset_mapping']
        }
        return result

    def _infer_sb_pos_ids(self):
        sb_pos_ids = []
        for i, (start, end) in self.sentence_b['aspect_term_mapping'].items():
            sb_pos_ids.extend(list(range(end - start)))
        return sb_pos_ids

    def generate_attention_mask(self, attention_mask, sb_len, mask_sentence_b=False):
        if not mask_sentence_b:
            return attention_mask + sb_len * [1]

        sa_len = len(attention_mask)
        sa_mask = np.array(attention_mask + sb_len * [0])[np.newaxis, :]
        sa_mask = np.repeat(sa_mask, sa_len, axis=0)  # (sa_len, sa_len + sb_len)

        left_mask = np.repeat(np.array(attention_mask)[np.newaxis, :], sb_len, axis=0)
        right_mask = np.zeros((sb_len, sb_len), dtype=int)
        for i, (start, end) in self.sentence_b['aspect_term_mapping'].items():
            index_grid = np.ix_(list(range(start, end)), list(range(start, end)))
            right_mask[index_grid] = np.ones_like(right_mask[index_grid], dtype=int)
        sb_mask = np.hstack([left_mask, right_mask])
        mask = np.vstack([sa_mask, sb_mask])
        mask[-1][sa_len:] = np.ones(sb_len)
        return mask


class PreTrainDataset(object):

    def __init__(self, file_path, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.data = self._load_data()

    
    def _load_data(self):
        result = []
        if self.file_path.endswith(".csv"):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".json"):
            data = json.load(self.file_path)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Wrong data format")
        
        return df

    def generate_string_sample(self):
        for idx, sample in self.data.iterrows():
            yield sample["context"], sample["topic"], sample["label"]

    def map_batch_string_to_tensor(self, context, topic, label):
        batch_size = context.shape[0]
        origin_context = [text.decode('utf-8') for text in context.numpy()]
        tokenized_context = self.tokenizer(
            origin_context,
            padding='longest',
            max_length=256,
            truncation=True
        )

        origin_topic = [text.decode('utf-8') for text in topic.numpy()]
        tokenized_topic = self.tokenizer(
            origin_topic,
            padding='longest',
            max_length=256,
            truncation=True
        )


        context_inputs_list = [
            np.array(tokenized_context['input_ids']),
            np.array(tokenized_context['token_type_ids']),
            np.array(tokenized_context["attention_mask"]),
            np.array(label).reshape((batch_size, -1))
        ]

        topic_inputs_list = [
            np.array(tokenized_topic['input_ids']),
            np.array(tokenized_topic['token_type_ids']),
            np.array(tokenized_topic['attention_mask'])
        ]

        return context_inputs_list + topic_inputs_list

    def __len__(self):
        return len(self.data)
    def wrap_map(self, context, topic, label):
        result = tf.py_function(
            self.map_batch_string_to_tensor,
            inp=[context, topic, label],
            Tout=[tf.int32] * 7
        )
        return result

if __name__ == '__main__':
    import os
    from transformers import AutoTokenizer
    from itertools import chain
    # file_path = ['data/Laptop-ACOS/processed_data/laptop_quad_train.tsv', 'data/Laptop-ACOS/processed_data/laptop_quad_dev.tsv', 'data/Laptop-ACOS/processed_data/laptop_quad_test.tsv']
    file_path = ['data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml', 'data/semeval2016/EN_REST_SB1_TEST_LABELED.xml']
    # file_path = ['data/semeval2015/ABSA-15_Restaurants_Train_Final.xml', 'data/semeval2015/ABSA15_Restaurants_Test.xml']
    model_name = 'microsoft/deberta-base'
    # model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f'bert_models/{os.path.basename(model_name)}')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer = None
    for fp in file_path:
        print(fp)
        all_cls_labels = []
        # dataset = ACOSDataset(fp, tokenizer, sentence_b=ACOS_LAPTOP_LABEL_MAPPING, model_type="end_to_end", tagging_schema="BIO")
        # dataset = EnglishDataset(fp, tokenizer, sentence_b=RES1516_LABEL_MAPPING, model_type="end_to_end", tagging_schema="BIO")
        # all_tuples = list(chain(*(json.loads(each[1]) for each in list(dataset.string_sentences))))
        # all_sentences = list(dataset.string_sentences)
        # n_implicit_tuples = sum([1 if each['target'] == "NULL" else 0 for each in all_tuples])
        # n_mixed_sentences = 0
        # for sent in all_sentences:
        #     sent_jsons = json.loads(sent[1])
        #     a_s_pairs = set((sent_json["category"], sent_json["polarity"]) for sent_json in sent_jsons)
        #     if len(a_s_pairs) > 1:
        #         n_mixed_sentences += 1 
        # print(len(all_sentences), n_mixed_sentences, n_mixed_sentences / len(all_sentences), len(all_tuples), n_implicit_tuples, n_implicit_tuples / len(all_tuples))
        
        dataset = EnglishDataset(fp, tokenizer, sentence_b=RES1516_LABEL_MAPPING, model_type="end_to_end", tagging_schema="BIO")
        ds = tf.data.Dataset.from_generator(
            dataset.generate_string_sample,
            output_types=(tf.string, tf.string)
        )
        # bd = ds.batch(batch_size=8).map(dataset.wrap_map)
        for a, b in ds.batch(16):
            cls_labels = dataset.map_batch_string_to_tensor_end_to_end(a, b)[4]
            all_cls_labels.append(cls_labels)
        # all_cls_labels = np.concatenate(all_cls_labels, axis=0)
        # print(all_cls_labels.shape)
        # p = all_cls_labels.sum() / all_cls_labels.size
        # print("success to batch ", fp, p)
    # input_ids = tokenizer(SENTENCE_B['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
    # tt = tokenizer.convert_ids_to_tokens(input_ids)
    
    # testtokenizer = TestTokenizer(tokenizer, SENTENCE_B)
    # testtokenizer.tokenize(['I like the fish!'])
    # pt = PreTrainDataset("data/amazon_review_processed.csv", tokenizer=tokenizer)
    # ds = tf.data.Dataset.from_generator(
    #     pt.data_generator,
    #     output_types=(tf.string, tf.string, tf.int32)
    # )
    # for a, b, c in ds.batch(8):
    #     pt.map_batch_string_to_tensor(a, b, c)
