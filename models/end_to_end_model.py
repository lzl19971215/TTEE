import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow_addons.text import crf_log_likelihood
from transformers.models.bert.modeling_tf_bert import TFBertAttention
from transformers import BertConfig


class BuildCRF(tfa.layers.CRF):

    def build(self, input_shape):
        # self._dense_layer.build(input_shape)
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

    def __init__(self, hidden_size, num_classes=5, dropout=0.3, block_output_activation=None, block_inter_activation="relu", **kwargs):
        super(TargetExtractionBlock, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        # self.linear_transform = keras.layers.Dense(hidden_size, activation='relu')
        transform_layers = []
        if hidden_size > 0:
            transform_layers.extend([keras.layers.Dense(hidden_size, activation=block_inter_activation), keras.layers.Dropout(dropout)])
        transform_layers.append(keras.layers.Dense(num_classes, activation=block_output_activation))
        self.linear_transform = keras.Sequential(transform_layers)
        self.crf = BuildCRF(units=num_classes, use_kernel=False)
        self.dropout = layers.Dropout(rate=dropout)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, hidden_states, mask=None, training=False, **kwargs):
        """

        :param hidden_states: (batch_size, num_aspects, seq_len, hidden_size)
        :param mask: crf mask
        :param kwargs: other param
        :return: ner logit, ner_loss
        """
        seq_len = tf.shape(hidden_states)[-2]
        hidden_size = tf.shape(hidden_states)[-1]
        hidden_states = tf.reshape(hidden_states, (-1, seq_len, hidden_size))
        h = self.linear_transform(hidden_states)
        h = self.dropout(h, training=training)
        decoded_sequence, output_logits, seq_length, chain_kernel = self.crf(h, mask=mask)
        return decoded_sequence, output_logits, seq_length, chain_kernel

    def compute_crf_loss(self, decoded_sequences, potentials, labels, sequence_length):
        labels = tf.reshape(labels, tf.shape(decoded_sequences))
        labels = tf.where(labels == -1, 0, labels)
        loss = -crf_log_likelihood(potentials, labels, sequence_length, self.crf.chain_kernel)[0]
        loss = tf.reduce_mean(loss / tf.cast(sequence_length, loss.dtype))
        y_pred_res = tf.cast(decoded_sequences == labels, tf.float32)
        max_len = tf.shape(y_pred_res)[1]
        mask = tf.sequence_mask(sequence_length, maxlen=max_len)
        y_pred_res = tf.where(mask, y_pred_res, 0)
        ave_acc = tf.reduce_sum(y_pred_res) / tf.cast(tf.reduce_sum(sequence_length), tf.float32)
        return loss, ave_acc

    # def build(self, input_shape):
    #     linear_input_shape = (None, input_shape[-1] * 2)
    #     self.linear_transform.build(linear_input_shape)
    #     self.crf.build((None, self.hidden_size))
    #     super(TargetExtractionBlock, self).build(input_shape)


class FuseNet(Layer):

    def __init__(self, d_model, fuse_strategy='concat', dropout=0.1):
        super(FuseNet, self).__init__()
        self.fuse_strategy = fuse_strategy
        self.d_model = d_model
        if self.fuse_strategy == "gate":
            self.context_w = keras.layers.Dense(d_model, activation="sigmoid")
            self.aspect_w = keras.layers.Dense(d_model, activation="sigmoid")
        self.dropout = dropout

    def call(self, inputs, training=False, output_sim=False, cross=True, **kwargs):
        """


        :param inputs: text_tokens (batch_size, seq_len, hidden_size); aspect_tokens (num_aspects, hidden_size)
        :param training:
        :param output_sim: whether to output sim_matrix
        :param kwargs:
        :return:
        """

        # get shape
        text_tokens, aspect_tokens = inputs
        batch_size = tf.shape(text_tokens)[0]
        seq_len = tf.shape(text_tokens)[1]
        num_aspects = tf.shape(aspect_tokens)[0]
        if cross:
            # (batch_size, num_aspects, seq_len, hidden_size)
            text_tokens = tf.repeat(text_tokens[:, tf.newaxis, :, :], num_aspects, axis=1)

            # (batch_size, num_aspects, 1, hidden_size)
            aspect_tokens = aspect_tokens[tf.newaxis, :, tf.newaxis, :]
        else:
            # (batch_size, 1, hidden_size)
            aspect_tokens = aspect_tokens[:, tf.newaxis, :]

        sim_matrix = self.cosine_similarity(aspect_tokens, text_tokens)

        # (batch_size, num_aspects, seq_len, hidden_size)
        aspect_text_tokens = text_tokens * sim_matrix
        aspect_tokens = tf.tile(aspect_tokens, [batch_size, 1, seq_len, 1])
        if training:
            aspect_text_tokens = tf.nn.dropout(aspect_text_tokens, rate=self.dropout)

        if self.fuse_strategy == 'concat':
            # output = tf.concat([text_tokens, aspect_text_tokens], axis=-1)
            output = tf.concat([aspect_text_tokens, aspect_tokens], axis=-1)
        elif self.fuse_strategy == 'add':
            output = text_tokens + aspect_text_tokens
        elif self.fuse_strategy == 'update':
            output = aspect_text_tokens
        elif self.fuse_strategy == "gate":
            context_f = self.context_w(text_tokens)
            aspect_i = self.aspect_w(aspect_text_tokens)
            output = text_tokens * context_f + aspect_text_tokens * aspect_i
        else:
            raise ValueError('Unsupported strategy {}'.format(self.fuse_strategy))
        result = {
            "fused_states": output
        }
        if output_sim:
            result["sim_matrix"] = sim_matrix
        return result

    @staticmethod
    def cosine_similarity(query, key):
        """

        :param query: (batch_size, query_len, 1, hidden_size)
        :param key: (batch_size, query_len, key_len, hidden_size)
        :return: sim_matrix: (batch_size, query_len, key_len, 1) value in (-1, 1)
        """

        query_mod = tf.sqrt(tf.reduce_sum(query * query, axis=-1, keepdims=True))
        key_mod = tf.sqrt(tf.reduce_sum(key * key, axis=-1, keepdims=True))
        sim_matrix = tf.reduce_sum(query * key, axis=-1, keepdims=True)
        # sim_matrix = sim_matrix / (query_mod * key_mod) + 1e-8
        sim_matrix = sim_matrix / (query_mod * key_mod + 1e-5)
        return sim_matrix


class End2EndAspectSentimentModel(Model):

    def __init__(
            self,
            init_bert_model,
            sentence_b,
            num_sentiment_classes=3,
            subblock_hidden_size=256,
            block_output_activation=None,
            block_inter_activation="relu",
            subblock_head_num=1,
            cache_dir=None,
            fuse_strategy='concat',
            pooling='cls',
            tagging_schema="BIOES",
            detect_loss="ce",
            extra_attention=True,
            hot_attention=False,
            do_logit_adjust=False,
            dropout=0.1,
            detect_dropout=0.1,
            loss_ratio=1.0,
            detect_label_prior=1,
            tau=1,
            **kwargs
    ):
        super(End2EndAspectSentimentModel, self).__init__()
        self.dropout = dropout
        self.bert = TFAutoModel.from_pretrained(init_bert_model, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(init_bert_model, cache_dir=cache_dir)
        self.fuse_net = FuseNet(self.bert.config.hidden_size, fuse_strategy, dropout=dropout)
        n_te_classes = 5 if tagging_schema == "BIOES" else 3
        self.te_block = TargetExtractionBlock(hidden_size=subblock_hidden_size, num_classes=n_te_classes, dropout=dropout, block_output_activation=block_output_activation, block_inter_activation=block_inter_activation)
        if fuse_strategy == 'concat':
            attention_hidden_size = self.bert.config.hidden_size * 2
        else:
            attention_hidden_size = self.bert.config.hidden_size

        self.extra_attention = extra_attention
        if self.extra_attention:
            attention_config = BertConfig(hidden_size=attention_hidden_size, num_attention_heads=self.bert.config.num_attention_heads)
            self.self_attention = TFBertAttention(attention_config)
            if hot_attention:
                # build layer with dummy input
                self.self_attention(tf.ones([1,10,attention_hidden_size], dtype=float), attention_mask=None, head_mask=None, output_attentions=False)
                # set weights equal to last self-attention in bert
                self.self_attention.set_weights(self.bert.bert.encoder.layer[-1].attention.get_weights())
        else:
            self.self_attention = None
        
        layers = []
        if subblock_hidden_size > 0:
            layers.extend([keras.layers.Dense(subblock_hidden_size, activation=block_inter_activation), keras.layers.Dropout(detect_dropout)])
        layers.append(keras.layers.Dense(2, activation=block_output_activation))
        self.contain_dense = keras.Sequential(layers)
        # self.contain_dense = keras.Sequential([
        #     keras.layers.Dense(subblock_hidden_size, activation='relu'),
        #     keras.layers.Dropout(0.2),
        #     keras.layers.Dense(1)
        # ])
        self.te_loss = self.te_block.compute_crf_loss
        # 
        assert detect_loss in ['ce', 'pwm', 'focal']
        if detect_loss == 'ce':
            self.detect_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif detect_loss == 'focal':
            self.detect_loss = self.focal_loss
        else:
            self.detect_loss = self.pair_wise_margin_loss
        self.config = {
            "init_bert_model": init_bert_model,
            "sentence_b": sentence_b,
            "num_sentiment_classes": num_sentiment_classes,
            "subblock_hidden_size": subblock_hidden_size,
            "subblock_head_num": subblock_head_num,
            "block_output_activation": block_output_activation,
            "block_inter_activation": block_inter_activation,
            "cache_dir": cache_dir,
            "fuse_strategy": fuse_strategy,
            "pooling": pooling,
            "tagging_schema": tagging_schema,
            "extra_attention": extra_attention,
            "hot_attention": hot_attention,
            "dropout": dropout,
            "detect_dropout": detect_dropout,
            "loss_ratio": loss_ratio,
            "detect_loss": detect_loss,
            "do_logit_adjust": do_logit_adjust,
            "detect_label_prior": detect_label_prior,
            "tau": tau
        }
        self.fuse_strategy = fuse_strategy
        self.pooling = pooling
        self.num_aspect_senti = len(sentence_b["texts"]) * len(sentence_b["sentiments"])
        self.d_model =  768 if 'base' in init_bert_model else 1024
        self.asp_senti_cache = tf.Variable(tf.zeros((self.num_aspect_senti, self.d_model)), trainable=False)
        # self.context_cache = tf.Variable(tf.zeros((train_batch_size, 512, self.d_model)), trainable=False)
        self.updated = tf.Variable(initial_value=False, dtype=tf.bool, trainable=False)
        # self.alpha = loss_ratio if loss_ratio > 1 else 1
        # self.beta = 1 if loss_ratio > 1 else 1 / loss_ratio
        self.alpha = loss_ratio /  (loss_ratio + 1) * 2
        self.beta = 1 / (loss_ratio + 1) * 2
        self.do_logit_adjust = do_logit_adjust
        self.tau = tau
        self.log_prior = tf.cast(tf.math.log(detect_label_prior), tf.float32) if self.do_logit_adjust or detect_loss == 'pwm' else -1
        print(self.log_prior)
        print(self.alpha)
        print(self.beta)
        # self.prior = 0.0034261559638023666

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self.config

    @tf.function
    def call(
            self,
            text_inputs,
            aspect_inputs,
            cache_text_states,
            label_inputs=None,
            phase="train",
            asp_senti_batch_idx=0,
            output_attentions=False,
    ):
        training = phase == "train"
        input_ids, token_type_ids, attention_mask = text_inputs
        # get bert output
        # if asp_senti_batch_idx == 0:
        #     bert_output = self.bert(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         token_type_ids=token_type_ids,
        #         training=(phase == "train")
        #     )
        #     text_states, _ = bert_output.last_hidden_state, bert_output.pooler_output
        #     self.context_cache[:tf.shape(text_states)[0], :tf.shape(text_states)[1]].assign(text_states)
        # else:
        #     text_states = self.context_cache[:tf.shape(input_ids)[0], :tf.shape(input_ids)[1]]
        flag = tf.reduce_sum(cache_text_states)
        # tf.print("flag: ", flag)
        if flag == 0:
            bert_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                training=(phase == "train")
            )
            # tf.print("re calculate")
            text_states, _ = bert_output.last_hidden_state, bert_output.pooler_output
        else:
            # tf.print("use cache")
            text_states = cache_text_states
        text_states.set_shape((None, None, self.d_model))

        if phase == "train":
            asp_senti_output = self.bert(
                input_ids=aspect_inputs[0],
                token_type_ids=aspect_inputs[1],
                attention_mask=aspect_inputs[2],
                training=True
            )
            asp_senti_cls_states = asp_senti_output.pooler_output
            # self.asp_senti_cache.scatter_update(tf.IndexedSlices(asp_senti_cls_states, tf.range(asp_senti_batch_idx, asp_senti_batch_idx + tf.shape(asp_senti_cls_states)[0])))
        elif phase == "pretrain":
            asp_senti_output = self.bert(
                input_ids=aspect_inputs[0],
                token_type_ids=aspect_inputs[1],
                attention_mask=aspect_inputs[2],
                training=True
            )
            asp_senti_cls_states = asp_senti_output.pooler_output
        elif phase == "dynamic_aspect_test":
            asp_senti_output = self.bert(
                input_ids=aspect_inputs[0],
                token_type_ids=aspect_inputs[1],
                attention_mask=aspect_inputs[2],
                training=False
            )
            asp_senti_cls_states = asp_senti_output.pooler_output            
        elif (phase == "test" or phase == "valid") and not self.updated:
            asp_senti_output = self.bert(
                input_ids=aspect_inputs[0],
                token_type_ids=aspect_inputs[1],
                attention_mask=aspect_inputs[2],
                training=False
            )
            asp_senti_cls_states = asp_senti_output.pooler_output
            self.asp_senti_cache.scatter_update(tf.IndexedSlices(asp_senti_cls_states, tf.range(asp_senti_batch_idx, asp_senti_batch_idx + tf.shape(aspect_inputs[0])[0])))
            if asp_senti_batch_idx + tf.shape(aspect_inputs[0])[0] == self.num_aspect_senti:
                self.updated.assign(True)
        else:
            asp_senti_cls_states = tf.nn.embedding_lookup(self.asp_senti_cache, tf.range(asp_senti_batch_idx, asp_senti_batch_idx + tf.shape(aspect_inputs[0])[0]))

        # fuse texts aspects

        # (batch_size, num_aspect_senti_pairs, seq_len, hidden_size * 2)
        fused_results = self.fuse_net(
            inputs=[text_states, asp_senti_cls_states], training=training, output_sim=output_attentions, cross=phase != "pretrain"
        )
        fused_states = fused_results["fused_states"]
        if output_attentions:
            sim_matrix = fused_results["sim_matrix"]

        if phase == "pretrain":
            # get shape
            batch_size = tf.shape(fused_states)[0]
            seq_len = tf.shape(fused_states)[1]
            hidden_size = tf.shape(fused_states)[2]

            # fused_states: (batch_size, seq_len, hidden_size)
            crf_mask = input_ids != 0
        else:
            # get shape
            batch_size = tf.shape(fused_states)[0]
            num_as_pairs = tf.shape(fused_states)[1]
            seq_len = tf.shape(fused_states)[2]
            hidden_size = tf.shape(fused_states)[3]

            # (batch_size * num_aspect_senti_pairs, seq_len, hidden_size * 2)
            fused_states = tf.reshape(fused_states, (-1, seq_len, hidden_size))
            # (batch_size * num_as_pairs, seq_len)

            crf_mask = input_ids != 0
            crf_mask = tf.tile(crf_mask[:, tf.newaxis, :], [1, num_as_pairs, 1])
            crf_mask = tf.reshape(crf_mask, (-1, seq_len))
        self_attention_mask = tf.cast(crf_mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]
        self_attention_mask = (1 - self_attention_mask) * -10000.0

        # attention test
        if self.extra_attention:
            output_states = self.self_attention(
                input_tensor=fused_states,
                attention_mask=self_attention_mask,
                head_mask=None,
                output_attentions=output_attentions
            )[0]
        else:
            output_states = fused_states

        # 3.15 add drop out
        if phase == "train" or phase == "pretrain":
            output_states = tf.nn.dropout(output_states, rate=self.dropout)

        if phase != "pretrain":
            # ner block
            decoded_sequence, output_logits, seq_length, chain_kernel = self.te_block(
                hidden_states=output_states,
                mask=crf_mask,
                training=training
            )
        if self.pooling == "cls":
            pool_states = output_states[:, 0, :]
        else:
            pool_states = tf.reduce_mean(output_states, axis=1)
        output_cls_states = self.contain_dense(pool_states)
        output_cls_states = tf.reshape(output_cls_states, (batch_size, -1, 2))  # (batch_size, num_as_pairs, 2)
        cls_probs = tf.nn.softmax(output_cls_states, -1)


        if phase == "train":
            ner_labels, cls_labels = label_inputs
            ner_loss, ner_acc = self.te_loss(decoded_sequence, output_logits, ner_labels, seq_length)
            cls_loss = self.detect_loss(cls_labels, output_cls_states)

            cls_predicts = tf.cast(cls_probs[:, :, 1] > 0.5, tf.int32)
            cls_acc = tf.reduce_mean(tf.cast(cls_labels == cls_predicts, tf.float32))
            cls_pos_acc = tf.reduce_sum(tf.cast((cls_labels == cls_predicts) & (cls_labels == 1), tf.float32)) / tf.reduce_sum(tf.cast(cls_labels == 1, tf.float32))
            cls_neg_acc = tf.reduce_sum(tf.cast((cls_labels == cls_predicts) & (cls_labels == 0), tf.float32)) / tf.reduce_sum(tf.cast(cls_labels == 0, tf.float32))
            total_loss = self.alpha * ner_loss + self.beta * cls_loss
            return [ner_loss, cls_loss, total_loss], [ner_acc, cls_acc, cls_pos_acc, cls_neg_acc], text_states
        elif phase == "pretrain":
            cls_labels = label_inputs[0]
            cls_loss = self.detect_loss(cls_labels, output_cls_states)
            cls_predicts = tf.cast(cls_probs[:, :, 1] > 0.5, tf.int32)
            cls_acc = tf.reduce_mean(tf.cast(cls_labels == cls_predicts, tf.float32))
            result = {
                "cls_loss": cls_loss,
                "cls_acc": cls_acc
            }
            return result
        elif phase == "valid":
            ner_labels, cls_labels = label_inputs
            ner_loss, ner_acc = self.te_loss(decoded_sequence, output_logits, ner_labels, seq_length)
            # apply post-hoc logit adjustment
            if self.do_logit_adjust:
                output_cls_states = self.logit_adjustment(output_cls_states)
            cls_loss = self.detect_loss(cls_labels, output_cls_states)

            cls_predicts = tf.cast(cls_probs[:, :, 1] > 0.5, tf.int32)
            cls_acc = tf.reduce_mean(tf.cast(cls_labels == cls_predicts, tf.float32))
            cls_pos_acc = tf.reduce_sum(tf.cast((cls_labels == cls_predicts) & (cls_labels == 1), tf.float32)) / tf.reduce_sum(tf.cast(cls_labels == 1, tf.float32))
            cls_neg_acc = tf.reduce_sum(tf.cast((cls_labels == cls_predicts) & (cls_labels == 0), tf.float32)) / tf.reduce_sum(tf.cast(cls_labels == 0, tf.float32))
            total_loss = self.alpha * ner_loss + self.beta * cls_loss
            return [ner_loss, cls_loss, total_loss], [ner_acc, cls_acc, cls_pos_acc, cls_neg_acc], text_states
        else:
            decoded_sequence = tf.reshape(decoded_sequence, (batch_size, num_as_pairs, seq_len))
            output_logits = tf.reshape(output_logits, (batch_size, num_as_pairs, seq_len, -1))
            # apply post-hoc logit adjustment
            if self.do_logit_adjust:
                output_cls_states = self.logit_adjustment(output_cls_states)            
            result = {
                "decoded_sequence": decoded_sequence,
                "output_logits": output_logits,
                "output_cls_states": cls_probs[:, :, 1],
                "text_states": text_states
            }
            if output_attentions:
                result["sim_matrix"] = sim_matrix
            # return ((decoded_sequence, output_logits), output_cls_states)
            return result

    def build_params(self):
        dummy_text_inputs = [tf.constant([[10, 10, 10]]), tf.constant([[0, 0, 0]]), tf.constant([[1, 1, 1]])]
        dummy_aspect_inputs = [tf.constant([[10, 10, 10]] * self.num_aspect_senti), tf.constant([[0, 0, 0]]* self.num_aspect_senti), tf.constant([[1, 1, 1]] * self.num_aspect_senti)]
        text_states = tf.zeros((1, 3, self.d_model))
        self.call(dummy_text_inputs, dummy_aspect_inputs, cache_text_states=text_states, phase="test")
        self.built = True
        # super(AspectSentimentModel, self).build(input_shape)
    
    def logit_adjustment(self, logits):
        # logits (batch_size, num_as_pairs)
        return logits - self.tau * self.log_prior
    
    def pair_wise_margin_loss(self, labels, logits):
        shift_logits = logits + self.tau * self.log_prior
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, shift_logits)
        return tf.reduce_mean(loss)

    def focal_loss(self, labels, logits):
        gamma = 2
        probs = tf.nn.softmax(logits, axis=-1)
        labels = tf.one_hot(labels, depth=2)
        loss = -tf.reduce_sum(labels * ((1 - probs) ** gamma) * tf.math.log(probs), axis=-1)
        return tf.reduce_mean(loss)




if __name__ == '__main__':
    import time
    import sys
    sys.path.append(".")
    from my_datasets import PreTrainDataset, TestTokenizer
    from label_mappings import RES1516_LABEL_MAPPING

    init_dir = 'bert_models/bert-base-cased'
    init_model = 'bert-base-cased'
    file_path = '../data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', cache_dir=init_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # dataset = SemEvalDataSet(file_path, tokenizer, sentence_b=ASPECT_SENTENCE, model_type="end_to_end")
    # ds = tf.data.Dataset.from_generator(
    #     dataset.generate_string_sample,
    #     output_types=(tf.string, tf.string)
    # ).batch(batch_size=8).map(dataset.wrap_map)
    
    pt = PreTrainDataset("data/pretrain/pretrain_data.csv", tokenizer=tokenizer)
    ds = tf.data.Dataset.from_generator(
        pt.generate_string_sample,
        output_types=(tf.string, tf.string, tf.int32)
    )
    loader = ds.batch(4).map(pt.wrap_map)
    model = End2EndAspectSentimentModel(init_bert_model=init_model, sentence_b=RES1516_LABEL_MAPPING, cache_dir=init_dir, fuse_strategy="update", tagging_schema="BIO")
    for inputs in loader:
        context = inputs[:3]
        label = inputs[3:4]
        topic = inputs[4:]
        output = model(context, topic, label, phase="pretrain")
        break
