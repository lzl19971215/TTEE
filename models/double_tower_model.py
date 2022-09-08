import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow_addons.text import crf_log_likelihood
from utils.eval_utils import Result
from label_mappings import NER_LABEL_MAPPING


class BuildCRF(tfa.layers.CRF):

    def build(self, input_shape):
        self._dense_layer.build(input_shape)
        super(BuildCRF, self).build(input_shape)


class IgnoreValueLoss(object):
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, y_true, y_pred, ignore_value=-1):
        fake_label = tf.where(y_true == ignore_value, tf.zeros_like(y_true), y_true)
        loss_matrix = self.loss(fake_label, y_pred)
        loss_matrix = tf.where(y_true == ignore_value, tf.zeros_like(loss_matrix), loss_matrix)
        y_pred_label = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred_res = tf.cast(y_true == y_pred_label, tf.float32)
        ave_acc = tf.reduce_sum(y_pred_res) / tf.reduce_sum(tf.cast(y_true != ignore_value, tf.float32))
        ave_loss = tf.reduce_sum(loss_matrix) / tf.reduce_sum(tf.cast(y_true != ignore_value, tf.float32))
        return ave_loss, ave_acc


class TargetExtractionBlock(Layer):

    def __init__(self, hidden_size, num_classes=5, **kwargs):
        super(TargetExtractionBlock, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.linear_transform = keras.layers.Dense(hidden_size, activation='relu')
        self.crf = BuildCRF(units=num_classes)
        self.dropout = layers.Dropout(rate=0.3)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32),
    #         tf.TensorSpec(shape=(None, None), dtype=tf.bool),
    #         tf.TensorSpec(shape=(), dtype=tf.bool)
    #     ])
    def call(self, hidden_states, mask=None, training=False, **kwargs):
        """

        :param hidden_states: (batch_size, seq_len, hidden_size)
        :param mask: crf mask
        :param kwargs: other param
        :return: ner logit, ner_loss
        """
        h = self.linear_transform(hidden_states)
        h = self.dropout(h, training=training)
        decoded_sequence, output_logits, seq_length, chain_kernel = self.crf(h, mask=mask)  # crf chain_kernel 未更新
        return decoded_sequence, output_logits, seq_length, chain_kernel

    def compute_loss(self, label, logit, ignore_idx=-1):
        fake_label = tf.where(label == ignore_idx, tf.zeros_like(label), label)
        loss_matrix = self.loss_fn(fake_label, logit)
        loss_matrix = tf.where(label == ignore_idx, tf.zeros_like(loss_matrix), loss_matrix)
        return tf.reduce_sum(loss_matrix) / tf.reduce_sum(tf.cast(label != -1, tf.float32))

    def compute_crf_loss(self, decoded_sequences, potentials, labels, sequence_length):
        labels = tf.where(labels == -1, 0, labels)
        loss = -crf_log_likelihood(potentials, labels, sequence_length, self.crf.chain_kernel)[0]
        loss = tf.reduce_mean(loss / tf.cast(sequence_length, loss.dtype))
        y_pred_res = tf.cast(decoded_sequences == labels, tf.float32)
        max_len = tf.shape(y_pred_res)[1]
        mask = tf.sequence_mask(sequence_length, maxlen=max_len)
        y_pred_res = tf.where(mask, y_pred_res, 0)
        ave_acc = tf.reduce_sum(y_pred_res) / tf.cast(tf.reduce_sum(sequence_length), tf.float32)
        return loss, ave_acc

    def build(self, input_shape):
        self.linear_transform.build(input_shape)
        self.crf.build((None, self.hidden_size))
        super(TargetExtractionBlock, self).build(input_shape)


class TargetAspectClassificationBlock(Layer):

    def __init__(self, hidden_size, num_aspects, num_heads, **kwargs):
        super(TargetAspectClassificationBlock, self).__init__(**kwargs)
        assert hidden_size % num_heads == 0, "Error hidden size with num heads ({}, {})".format(hidden_size, num_heads)
        self.hidden_size = hidden_size
        self.linear_transform = keras.layers.Dense(self.hidden_size, activation='relu')
        self.classification_transform = keras.layers.Dense(1)
        self.attention_hidden_size = self.hidden_size // num_heads
        self.self_attention = SelfAttention(num_heads=num_heads, attention_head_size=self.attention_hidden_size)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.dropout = layers.Dropout(rate=0.3)
        self.num_aspects = num_aspects

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32),
    #         tf.TensorSpec(shape=(None, 12, 768), dtype=tf.float32),
    #         tf.TensorSpec(shape=(None, 14), dtype=tf.int32),
    #         tf.TensorSpec(shape=(None, ), dtype=tf.int32),
    #         tf.TensorSpec(shape=(None, None, None), dtype=tf.bool),
    #         tf.TensorSpec(shape=(), dtype=tf.bool),
    #         tf.TensorSpec(shape=(), dtype=tf.bool)
    #     ]
    # )
    def call(self, all_tokens, aspect_tokens, target_pos_info, target_batch_info, attention_mask,
             output_attentions=False, training=False, **kwargs):
        """
        :param all_tokens: (batch_size, seq_length, hidden_size)
        :param aspect_tokens: (batch_size, num_aspects, hidden_size)
        :param target_pos_info: (batch_num_targets,)
        :param target_batch_info: (batch_num_targets,)
        :param attention_mask: (batch_size, seq_length,)
        :param kwargs:
        :return:
        """
        target_slices = target_pos_info[:, :2]
        target_array, attention_scores = self.gather_tokens_hidden_state(all_tokens, target_slices, target_batch_info,
                                                                         attention_mask=attention_mask,
                                                                         output_attentions=output_attentions,
                                                                         use_attention=True, training=training)
        batch_size = tf.shape(all_tokens)[0]
        num_aspects = self.num_aspects

        h_a = self.dropout(self.linear_transform(aspect_tokens), training=training)

        t_a_tensor = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        for i in tf.range(batch_size):
            # (one_batch_num_target, hidden_size)
            # one_batch_target = tf.gather_nd(h_t, tf.where(batch_clf_index == i))
            batch_slices = tf.cast(tf.squeeze(tf.where(target_batch_info == i), axis=1), tf.int32)
            one_batch_target = target_array.gather(batch_slices)

            # (one_batch_num_target, 1, hidden_size)
            one_batch_target = tf.expand_dims(one_batch_target, 1)

            # (one_batch_num_target, num_aspects, hidden_size)
            one_batch_target = tf.repeat(one_batch_target, num_aspects, axis=1)

            # (one_batch_num_target, num_aspects, hidden_size)
            one_batch_aspects = tf.repeat(h_a[i: i + 1], tf.shape(one_batch_target)[0], axis=0)
            # (one_batch_num_target, num_aspects, hidden_size * 2)
            t_a_tensor = t_a_tensor.write(i, tf.concat([one_batch_target, one_batch_aspects], axis=-1))

        t_a_tensor = t_a_tensor.concat()
        t_a_tensor = tf.ensure_shape(t_a_tensor, (
        None, num_aspects, self.hidden_size * 2))  # (batch_num_target, num_aspects, hidden_size * 2)
        t_a_logit = self.classification_transform(t_a_tensor)
        return (t_a_logit, attention_scores)

    def build(self, input_shape):
        self.linear_transform.build(input_shape)
        self.self_attention.build(input_shape=(None, None, self.hidden_size))
        self.classification_transform.build((None, self.hidden_size * 2))
        super(TargetAspectClassificationBlock, self).build(input_shape)

    def gather_tokens_hidden_state(self, all_tokens, slices, batch_index, attention_mask=None, output_attentions=False,
                                   use_attention=False, training=False):
        """
        gather tokens rep from slices index
        :param all_tokens:  (batch_size, seq_len, hidden_size)
        :param slices:  (Any, 2)
        :param batch_index: (slices.shape[0], )
        :return:
        """
        num_slices = tf.shape(slices)[0]
        pool_tensors = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        all_attentions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        all_tokens = self.dropout(self.linear_transform(all_tokens), training=training)  # pre transform
        for i in tf.range(num_slices):
            batch_order = batch_index[i]
            query_tensors = all_tokens[batch_order: batch_order + 1, slices[i][0]: slices[i][1], :]
            key_tensors = all_tokens[batch_order: batch_order + 1]
            if use_attention:
                mask = attention_mask[batch_order: batch_order + 1, 0: 1]
                if not output_attentions:
                    target_tensors = self.self_attention(query_tensors, key_tensors,
                                                         attention_mask=mask)  # q, k transform
                else:
                    target_tensors, attention_scores = self.self_attention(query_tensors, key_tensors,
                                                                           attention_mask=mask,
                                                                           output_attentions=output_attentions)
                    all_attentions = all_attentions.write(i, attention_scores)
            else:
                target_tensors = query_tensors

            target_tensors = tf.reshape(target_tensors, (slices[i][1] - slices[i][0], -1))
            target_tensors = tf.reduce_mean(target_tensors, axis=0)
            pool_tensors = pool_tensors.write(i, target_tensors)

        return (pool_tensors, all_attentions)


class SentimentClassificationBlock(Layer):

    def __init__(self, hidden_size, num_classes, num_heads, **kwargs):
        super(SentimentClassificationBlock, self).__init__(**kwargs)
        assert hidden_size % num_heads == 0, "Error hidden size with num heads ({}, {})".format(hidden_size, num_heads)
        self.hidden_size = hidden_size
        self.linear_transform = keras.layers.Dense(hidden_size, activation='relu')
        self.linear_output = keras.layers.Dense(num_classes)
        self.attention_hidden_size = self.hidden_size // num_heads
        self.self_attention = SelfAttention(num_heads=num_heads, attention_head_size=self.attention_hidden_size)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.dropout = Dropout(rate=0.3)

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32),
    #         tf.TensorSpec(shape=(None, 12, 768), dtype=tf.float32),
    #         tf.TensorSpec(shape=(None, 4), dtype=tf.int32),
    #         tf.TensorSpec(shape=(None, ), dtype=tf.int32),
    #         tf.TensorSpec(shape=(None, None, None), dtype=tf.bool),
    #         tf.TensorSpec(shape=(), dtype=tf.bool),
    #         tf.TensorSpec(shape=(), dtype=tf.bool)
    #     ]
    # )
    def call(self, all_tokens, aspect_tokens, target_aspect_pos_info, target_aspect_batch_index, attention_mask,
             output_attentions=False, training=False, **kwargs):
        """[summary]

        Args:
            all_tokens ([type]): [batch_size, seq_len, hidden_size]
            aspect_tokens ([type]): [batch_size, num_aspects, hidden_size]
            target_aspect_pos_info ([type]): [num_pairs, 4]
            target_aspect_batch_index ([type]): [num_pairs, ]
            attention_mask ([type]): [batch_size, seq_len, seq_len]
            output_attentions (bool, optional): [True / False]. Defaults to False.
            training (bool, optional): [True / False]. Defaults to False.
            phase (str, optional): ["train" "valid" "test"]. Defaults to "train".

        Returns:
            [type]: [description]
        """
        target_pos_info = target_aspect_pos_info[:, :2]
        aspect_pos_info = target_aspect_pos_info[:, 2:4]

        t_a_pair_rep, attention_scores = self.gather_target_aspect_pair_hidden_state(
            all_tokens, aspect_tokens, target_pos_info, aspect_pos_info, target_aspect_batch_index, attention_mask,
            output_attentions=output_attentions, training=training
        )

        # (num_pairs, hidden_size)
        h_ta_logit = self.linear_output(t_a_pair_rep)
        return (h_ta_logit, attention_scores)

    def compute_loss(self, label, logit, ignore_idx=-1):
        fake_label = tf.where(label == ignore_idx, tf.zeros_like(label), label)
        loss_matrix = self.loss_fn(fake_label, logit)
        loss_matrix = tf.where(label == ignore_idx, tf.zeros_like(loss_matrix), loss_matrix)
        return tf.reduce_sum(loss_matrix) / tf.reduce_sum(tf.cast(label != -1, tf.float32))

    def build(self, input_shape):
        self.linear_transform.build(input_shape)
        self.self_attention.build((None, None, self.hidden_size))
        self.linear_output.build((None, self.hidden_size))

        super(SentimentClassificationBlock, self).build(input_shape)

    def gather_target_aspect_pair_hidden_state(self, all_tokens, aspect_tokens, target_slices, aspect_slices,
                                               target_aspect_batch_index, attention_mask=None, output_attentions=False,
                                               training=False):
        """
        gather tokens rep from slices index
        :param all_tokens:  (batch_size, seq_len, hidden_size)
        :param slices:  (Any, 2)
        :param batch_index: (slices.shape[0], )
        :return:
        """
        num_slices = tf.shape(target_slices)[0]
        pool_tensors = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        all_attentions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)

        all_tokens = self.dropout(self.linear_transform(all_tokens), training=training)
        aspect_tokens = self.dropout(self.linear_transform(aspect_tokens), training=training)

        for i in tf.range(num_slices):
            batch_order = target_aspect_batch_index[i]

            # (1, num_target_tokens, hidden_size)
            query_tensors = all_tokens[batch_order: batch_order + 1, target_slices[i][0]: target_slices[i][1], :]

            # (1, num_aspect_tokens, hidden_size)
            aspect_tensors = aspect_tokens[batch_order: batch_order + 1, aspect_slices[i][0]: aspect_slices[i][1], :]

            # (1, 1, hidden_size)
            pool_aspect_tensors = tf.reduce_mean(aspect_tensors, axis=1, keepdims=True)

            # (1, num_target_tokens, hidden_size)
            merge_tensors = query_tensors + pool_aspect_tensors

            key_tensors = all_tokens[batch_order: batch_order + 1]
            mask = attention_mask[batch_order: batch_order + 1, 0: 1]
            if not output_attentions:
                target_tensors = self.self_attention(merge_tensors, key_tensors, attention_mask=mask)
            else:
                target_tensors, attention_scores = self.self_attention(merge_tensors, key_tensors, attention_mask=mask,
                                                                       output_attentions=output_attentions)
                all_attentions = all_attentions.write(i, attention_scores)
            target_tensors = tf.reshape(target_tensors, (target_slices[i][1] - target_slices[i][0], -1))
            target_tensors = tf.reduce_mean(target_tensors, axis=0)
            pool_tensors = pool_tensors.write(i, target_tensors)

        pool_tensors = pool_tensors.stack()
        return (pool_tensors, all_attentions)


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, attention_head_size, **kwargs):
        super().__init__(**kwargs)

        self.num_attention_heads = num_heads
        self.attention_head_size = attention_head_size
        self.query = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_attention_heads, self.attention_head_size),
            bias_axes="de",
            name="query",
        )
        self.key = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_attention_heads, self.attention_head_size),
            bias_axes="de",
            name="key",
        )
        # self.value = tf.keras.layers.experimental.EinsumDense(
        #     equation="abc,cde->abde",
        #     output_shape=(None, self.num_attention_heads, self.attention_head_size),
        #     bias_axes="de",
        #     name="value",
        # )
        self.dropout = tf.keras.layers.Dropout(rate=0.3)

    def call(self, query_states, key_states, attention_mask=None, head_mask=None, output_attentions=False,
             training=False):
        query_layer = self.query(inputs=query_states)  # (batch_size, query_len, num_heads, hidden_size)
        key_layer = self.key(inputs=key_states)  # (batch_size, key_len, num_heads, hidden_size)
        # value_layer = self.value(inputs=key_states) # (batch_size, key_len, num_heads, hidden_size)
        value_layer = tf.reshape(key_states, tf.shape(key_layer))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        dk = tf.cast(x=self.attention_head_size, dtype=query_layer.dtype)
        query_layer = tf.multiply(x=query_layer, y=tf.math.rsqrt(x=dk))
        attention_scores = tf.einsum("aecd,abcd->acbe", key_layer,
                                     query_layer)  # (batch_size, num_heads, query_len, key_len)

        if attention_mask is not None:  # lzl
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_mask = tf.cast(attention_mask, attention_scores.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_scores = attention_scores * head_mask

        attention_output = tf.einsum("acbe,aecd->abcd", attention_probs,
                                     value_layer)  # (batch_size, query_len, num_heads, hidden_size)
        # outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return attention_output if not output_attentions else (attention_output, attention_probs)

    def build(self, input_shape):
        self.query.build(input_shape=input_shape)
        self.key.build(input_shape=input_shape)
        super(SelfAttention, self).build(input_shape=input_shape)


# noinspection PyCallingNonCallable
class DoubleTowerAspectSentimentModel(Model):

    def __init__(
            self,
            init_bert_model,
            sentence_b,
            num_sentiment_classes=3,
            subblock_hidden_size=256,
            subblock_head_num=1,
            cache_dir=None
    ):
        super(DoubleTowerAspectSentimentModel, self).__init__()
        self.bert = TFAutoModel.from_pretrained(init_bert_model, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(init_bert_model, cache_dir=cache_dir)
        self.sentence_b_idx_map = tf.convert_to_tensor(list(sentence_b['aspect_term_mapping'].values()))
        self.te_block = TargetExtractionBlock(hidden_size=subblock_hidden_size)
        self.ta_block = TargetAspectClassificationBlock(hidden_size=subblock_hidden_size,
                                                        num_aspects=len(self.sentence_b_idx_map),
                                                        num_heads=subblock_head_num)
        self.sc_block = SentimentClassificationBlock(hidden_size=subblock_hidden_size,
                                                     num_classes=num_sentiment_classes,
                                                     num_heads=subblock_head_num)
        self.target_aspect_pooling = tf.reduce_mean

        self.te_loss = self.te_block.compute_crf_loss
        self.ta_loss = keras.losses.BinaryCrossentropy(from_logits=True)

        self.sc_loss = IgnoreValueLoss(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none'))

        self.config = {
            "init_bert_model": init_bert_model,
            "sentence_b": sentence_b,
            "num_sentiment_classes": num_sentiment_classes,
            "subblock_hidden_size": subblock_hidden_size,
            "subblock_head_num": subblock_head_num,
            "cache_dir": cache_dir
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self.config


    def call(
            self,
            text_inputs,
            aspect_inputs,
            phase="train",
            output_attentions=False
    ):
        training = phase == "train"
        if phase == "train" or phase == "valid":
            assert len(text_inputs) == 9, "{}".format(len(text_inputs))
            input_ids, token_type_ids, position_ids, attention_mask, \
            ner_labels, classify_labels, sentiment_labels, batch_senti_index, batch_clf_index = text_inputs
            # assert ner_labels is not None and classify_labels is not None and sentiment_labels is not None and \
            #        batch_senti_index is not None and batch_clf_index is not None
        else:
            assert len(text_inputs) == 4, "{}".format(len(text_inputs))
            input_ids, token_type_ids, position_ids, attention_mask = text_inputs
            ner_labels, classify_labels, sentiment_labels, \
            batch_senti_index, batch_clf_index = None, None, None, None, None

        # get bert output
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=(phase == "train")
        )
        tokens_hidden_state, cls_state = bert_output.last_hidden_state, bert_output.pooler_output

        asp_output = self.bert(
            input_ids=aspect_inputs[0],
            token_type_ids=aspect_inputs[1],
            attention_mask=aspect_inputs[2],
            training=(phase == "train")
        )
        asp_cls_rep = asp_output.pooler_output
        # get shape
        batch_size = tf.shape(tokens_hidden_state)[0]
        seq_len = tf.shape(tokens_hidden_state)[1]
        hidden_size = tf.shape(tokens_hidden_state)[2]
        num_aspects = len(self.sentence_b_idx_map)
        batch_num_ta_clf_targets = tf.shape(tokens_hidden_state)[0]
        batch_num_ta_senti_targets = tf.shape(tokens_hidden_state)[0]
        # crf_mask = tf.concat([input_ids[:, :sb_start] != 0, tf.fill((batch_size, sb_len), False)], axis=1)
        crf_mask = input_ids != 0
        # (batch_size, num_aspects, hidden_size)

        asp_rep = tf.repeat(asp_cls_rep[tf.newaxis, ...], [batch_size], axis=0)
        # ner block
        decoded_sequence, ner_logits, seq_length, chain_kernel = self.te_block(hidden_states=tokens_hidden_state,
                                                                               mask=crf_mask, training=training)
        if phase == "train" or phase == "valid":
            ner_loss, ner_acc = self.te_loss(decoded_sequence, ner_logits, ner_labels, seq_length)
        else:

            ner_predicts = tf.argmax(ner_logits, axis=-1, output_type=tf.int32)
            ner_predicts = tf.where(tf.cast(attention_mask[:, 0, :], bool), ner_predicts, 0)
            ner_result = self.decode_ner_output(ner_predicts)
            # if tf.shape(ner_result)[0] == 0:
            #     return None
            batch_clf_index = ner_result[:, -1]
            classify_labels = ner_result[:, :2]
            # tf.print("Ner result: ", ner_result)

        # target_aspect_classification block
        # (batch_num_ta_clf_targets, hidden_size)
        target_clf_logits, ta_attentions = self.ta_block(
            all_tokens=tokens_hidden_state,
            aspect_tokens=asp_rep,
            target_pos_info=classify_labels,
            target_batch_info=batch_clf_index,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            training=training
        )

        if phase == "train" or phase == "valid":
            ta_clf_loss = self.ta_loss(y_true=classify_labels[:, 2:], y_pred=target_clf_logits)
            ta_clf_res = tf.cast(classify_labels[:, 2:] == tf.squeeze(tf.cast(target_clf_logits > 0, tf.int32)),
                                 tf.float32)
            ta_clf_acc = tf.reduce_mean(ta_clf_res)
            target_aspect_pos_info = sentiment_labels[:, :4]
            # sc_target_range = sentiment_labels[:, :2]
            # sc_aspect_range = sentiment_labels[:, 2:4]
        else:
            # tf.print(target_clf_logits, summarize=-1)
            ta_clf_predict = tf.squeeze(target_clf_logits > 0, axis=-1)  # change
            # tf.print("ta_clf_predict: ", ta_clf_predict)
            target_aspect_pair = self.generate_target_aspect_pair_for_inference(ta_clf_predict, ner_result)
            if target_aspect_pair is None:
                return None if not output_attentions else (None, (ta_attentions, None))
            batch_senti_index = target_aspect_pair[:, -1]
            target_aspect_pos_info = target_aspect_pair[:, :4]
            # sc_target_range = target_aspect_pair[:, :2]
            # sc_aspect_range = target_aspect_pair[:, 2:4]

        ta_senti_logits, sc_attentions = self.sc_block(
            all_tokens=tokens_hidden_state,
            aspect_tokens=asp_rep,
            target_aspect_pos_info=target_aspect_pos_info,
            target_aspect_batch_index=batch_senti_index,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            training=training
        )

        if phase == "train" or phase == "valid":
            ta_senti_loss, ta_senti_acc = self.sc_loss(y_true=sentiment_labels[:, -1], y_pred=ta_senti_logits,
                                                       ignore_value=-1)
            total_loss = ner_loss + ta_clf_loss + ta_senti_loss
            return [ner_loss, ta_clf_loss, ta_senti_loss, total_loss], [ner_acc, ta_clf_acc, ta_senti_acc]
        else:
            ta_senti_predict = tf.reshape(
                tf.argmax(ta_senti_logits, axis=-1, output_type=tf.int32),
                (tf.shape(ta_senti_logits)[0], 1)
            )
            output = tf.concat(
                [target_aspect_pos_info,
                 ta_senti_predict,
                 tf.reshape(batch_senti_index, tf.shape(ta_senti_predict))
                 ],
                axis=-1
            )
            return output if not output_attentions else (output, (ta_attentions, sc_attentions))

    def build_params(self, input_shape):
        self.te_block.build(input_shape)
        self.ta_block.build(input_shape)
        self.sc_block.build(input_shape)
        self.built = True
        # super(AspectSentimentModel, self).build(input_shape)

    @staticmethod
    def generate_target_aspect_pair_for_inference(ta_clf_predict, extract_target_index):
        target_aspect_pair = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, infer_shape=False)
        num_targets = tf.shape(ta_clf_predict)[0]
        write_index = 0
        for i in tf.range(num_targets):
            aspect_index = tf.cast(tf.where(ta_clf_predict[i]), dtype=tf.int32)
            aspect_range = tf.map_fn(lambda x: tf.concat([x, x + 1], axis=0), aspect_index)
            target_range = extract_target_index[i: i + 1, :2]
            if tf.shape(aspect_range)[0] == 0:
                if target_range[0, 0] == 0 and target_range[0, 1] == 1:
                    continue
                else:
                    aspect_range = tf.constant([[9, 10]], dtype=tf.int32)

            target_range = tf.repeat(target_range, tf.shape(aspect_range)[0], axis=0)
            batch_index = tf.repeat(extract_target_index[i: i + 1, -1:], tf.shape(aspect_range)[0], axis=0)
            # tf.print(aspect_range.shape, target_range.shape, batch_index.shape)
            target_aspect_pair = target_aspect_pair.write(
                write_index,
                tf.concat([target_range, aspect_range, batch_index], axis=-1)
            )
            write_index += 1
        if target_aspect_pair.size() == 0:
            return None
        else:
            return target_aspect_pair.concat()

    @staticmethod
    def _decode_start_end_for_ner_predict(tags):
        start_end_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        ta_index = 0
        tag_index = 0
        have_prefix = tf.constant(False)
        entity_start = 0
        for tag in tags[:-1]:
            if tag == NER_LABEL_MAPPING["S"]:
                if have_prefix:
                    # target_range = tf.concat([tf.constant([entity_start, tag_index + 1], dtype=tf.int32), tags[-1:]], 0)
                    target_range = tf.stack([entity_start, tag_index + 1, tags[-1]])
                    start_end_ta = start_end_ta.write(
                        ta_index,
                        target_range
                    )
                    have_prefix = tf.constant(False)
                else:
                    target_range = tf.stack([tag_index, tag_index + 1, tags[-1]])
                    # tf.concat([tf.constant([tag_index, tag_index + 1], dtype=tf.int32), tags[-1:]], 0)
                    start_end_ta = start_end_ta.write(
                        ta_index,
                        target_range
                    )
                ta_index += 1
            elif tag == NER_LABEL_MAPPING["B"]:
                if have_prefix:
                    # tf.concat([tf.constant([entity_start, tag_index], dtype=tf.int32), tags[-1:]], 0)
                    target_range = tf.stack([entity_start, tag_index, tags[-1]])
                    start_end_ta = start_end_ta.write(
                        ta_index,
                        target_range
                    )
                    ta_index += 1
                have_prefix = tf.constant(True)
                entity_start = tag_index
            elif tag == NER_LABEL_MAPPING["I"]:
                #  Model predict "IIIE" instead of "BIIE"
                if not have_prefix:
                    entity_start = tag_index
                    have_prefix = tf.constant(True)

            elif tag == NER_LABEL_MAPPING["O"]:
                if have_prefix:
                    # tf.concat([tf.constant([entity_start, tag_index], dtype=tf.int32), tags[-1:]], 0)
                    target_range = tf.stack([entity_start, tag_index, tags[-1]])
                    start_end_ta = start_end_ta.write(
                        ta_index,
                        target_range
                    )
                    ta_index += 1

                    have_prefix = tf.constant(False)
            else:    # E
                if have_prefix:
                    # tf.concat([tf.constant([entity_start, tag_index + 1], dtype=tf.int32), tags[-1:]], 0)
                    target_range = tf.stack([entity_start, tag_index + 1, tags[-1]])
                    start_end_ta = start_end_ta.write(
                        ta_index,
                        target_range
                    )
                    ta_index += 1

                    have_prefix = tf.constant(False)
            tag_index += 1

        if have_prefix:
            # tf.concat([tf.constant([entity_start, tag_index], dtype=tf.int32), tags[-1:]], 0)
            target_range = tf.stack([entity_start, tag_index, tags[-1]])
            start_end_ta = start_end_ta.write(
                ta_index,
                target_range
            )
            # self.append(entity_name, entity_start, idx)
        return start_end_ta.stack()

    def decode_ner_output(self, ner_predicts):
        batch_size = tf.shape(ner_predicts)[0]
        ner_predicts_with_batch_index = tf.concat(
            [ner_predicts, tf.reshape(tf.range(batch_size), (-1, 1))],
            axis=1
        )
        res = tf.TensorArray(dtype=tf.int32, size=0, infer_shape=False, dynamic_size=True)
        write_index = 0
        for i in tf.range(batch_size):
            #  write cls
            cls_tensor = tf.concat([tf.constant([0, 1]), i[tf.newaxis,]], axis=0)[tf.newaxis,]
            res = res.write(write_index, cls_tensor)
            write_index += 1
            write_tensor = self._decode_start_end_for_ner_predict(ner_predicts_with_batch_index[i])
            if tf.shape(write_tensor)[0] != 0:
                res = res.write(write_index, write_tensor)
                write_index += 1
        return res.concat()


if __name__ == '__main__':
    import time
    from datasets import SemEvalDataSet, TestTokenizer
    from label_mappings import SENTENCE_B
    from tensorflow.keras import backend

    init_dir = '/lzl/models/bert-base-cased'
    init_model = 'bert-base-cased'
    file_path = 'data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', cache_dir=init_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = SemEvalDataSet(file_path=file_path, tokenizer=tokenizer, sentence_b=SENTENCE_B)
    model = AspectSentimentModel(init_bert_model=init_model, aspect_term_index=None, cache_dir=init_dir,
                                 sentence_b=SENTENCE_B)
    # input_shape = [
    #             (None, None),
    #             (None, None),
    #             (None, None),
    #             (None, None),
    #             (None, None),
    #             (None, 14),
    #             (None, 5),
    #             (None, ),
    #             (None, ),
    # ]
    # if tf.executing_eagerly():
    #     graph = tf.__internal__.FuncGraph('build_graph')
    # else:
    #     graph = backend.get_graph()
    # with graph.as_default():
    #     input_shape = [tf.compat.v1.placeholder(shape=shape, dtype=tf.int32) for shape in input_shape]
