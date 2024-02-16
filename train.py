# -*- coding: utf-8 -*-
from multiprocessing import context
import os
import time
import tensorflow as tf
import logging
import argparse
import json
import random
import numpy as np
from models.single_tower_model import AspectSentimentModel
from models.double_tower_model import DoubleTowerAspectSentimentModel
from models.end_to_end_model import End2EndAspectSentimentModel
from my_datasets import ChineseDataset, EnglishDataset, PreTrainDataset, ACOSDataset
from label_mappings import *
from transformers import AutoTokenizer
from utils.data_utils import prepare_logger
from utils.eval_utils import Result, Triplet, compute_f1, save_predict
from collections import namedtuple
from tqdm import tqdm


def set_seed(seed=1):
    tf.random.set_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="训练任务名称", required=True, type=str)
    parser.add_argument("--init_model_dir", help="热启动模型路径", default="", type=str)
    parser.add_argument("--load_config", help="是否需要加载checkpoint的config", default=False, action="store_true")
    parser.add_argument("--save_dir", help="模型保存路径", default="", type=str)
    parser.add_argument("--output_dir", help="日志及测试结果保存路径", default="", type=str)
    parser.add_argument("--data_aug", help="数据增强策略", default=None, type=int)
    parser.add_argument("--dataset", help="数据集", default="res16", type=str)
    parser.add_argument("--train_batch_size", help="训练batch size", default=16, type=int)
    parser.add_argument("--neg_sample", help="采样方面-情感数量", default=-1, type=int)
    parser.add_argument("--cache_train_loader", help="是否固定采样的数据", default=False, action="store_true")
    parser.add_argument("--data_sample_ratio", help="训练样本采样比率", default=-1, type=float)
    parser.add_argument("--aspect_senti_batch_size", help="方面情感组合的batch_size(当组合数量过多时使用,防止OOM)", default=-1, type=int)
    parser.add_argument("--aspect_senti_test_batch_size", help="方面情感组合的batch_size(当组合数量过多时使用,防止OOM)", default=-1, type=int)
    parser.add_argument("--test_batch_size", help="测试batch size", default=32, type=int)
    parser.add_argument("--epochs", help="训练epoch数量", default=30, type=int)
    parser.add_argument("--valid_freq", help="验证的频率", default=1, type=int)
    parser.add_argument("--test_freq", help="测试的频率", default=1, type=int)
    parser.add_argument("--lr", help="学习率", default=1e-5, type=float)
    parser.add_argument("--decay_steps", default=-1, type=int)
    parser.add_argument("--decay_rate", default=0.9, type=float)
    parser.add_argument("--dropout_rate", help="失活率", default=0.1, type=float)
    parser.add_argument("--detect_dropout_rate", help="detect失活率", default=0.1, type=float)
    parser.add_argument("--block_att_head_num", help="子模块自注意力头数", default=1, type=int)
    parser.add_argument("--fuse_strategy", help="端到端的FuseNet融合策略", default="update", type=str)
    parser.add_argument("--pooling", help="池化策略:cls或者mean", default="cls", type=str)
    parser.add_argument("--schema", help="NER标注规则:BIOES/BIO", default="BIOES", type=str)    
    parser.add_argument("--extra_attention", help="端到端FuseNet之后是否加self Attention", default=False, action="store_true")
    parser.add_argument("--hot_attention", help="是否用bert最后一个self-attention参数初始化extra-attention", default=False, action="store_true")
    parser.add_argument("--d_block", help="子模块模型维度", default=256, type=int)
    parser.add_argument("--block_inter_activation", help="子模块中间层activation", default="relu", type=str)
    parser.add_argument("--block_output_activation", help="子模块输出activation", default=None, type=str)
    parser.add_argument("--model_type", help="模型种类（单塔、双塔、端到端）", default="end_to_end", type=str)
    parser.add_argument("--mask_sb", help="单塔模型attention mask, 不看sentence b", default=False, action="store_true")
    parser.add_argument("--cased", help="模型是否区分大小写", default=0, type=int)
    parser.add_argument("--detect_loss", help="ce, focal, pwm", default="ce", type=str)
    parser.add_argument("--logit_adjust", help="是否在非训练时进行logit 调整（解决样本不平衡）", default=False, action="store_true")
    parser.add_argument("--tau", help="logit adjust 超参数", default=1.0, type=float)          
    parser.add_argument("--do_train", help="是否进行训练", default=False, action="store_true")
    parser.add_argument("--do_valid", help="是否进行验证", default=False, action="store_true")
    parser.add_argument("--do_test", help="是否进行测试", default=False, action="store_true")
    parser.add_argument("--language", help="语言", default="en", type=str)
    parser.add_argument("--bert_size", help="预训练模型大小", default="base", type=str)
    parser.add_argument("--drop_null_data", help="是否在训练与测试集中去掉不包含三元组的句子", default=False, action="store_true")
    parser.add_argument("--loss_ratio", help="ner loss与detection loss的比例", default=1, type=float)

    # for pretrain
    parser.add_argument("--pretrain", help="是否是预训练", default=False, action="store_true")
    parser.add_argument("--pretrain_steps", help="预训练batch数量", type=int)
    parser.add_argument("--pretrain_save_steps", help="预训练日志打印步数", type=int)    
    parser.add_argument("--pretrain_log_steps", help="预训练模型保存步数", type=int)
    parser.add_argument("--pretrain_data", help="预训练数据文件名称", type=str)

    # for speed test
    parser.add_argument("--speed_test", help="进行speed test", default=False, action="store_true")
    return parser.parse_args()

args = arg_parse()

single_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, 14), dtype=tf.int32),
    tf.TensorSpec(shape=(None, 5), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32),
]

aspect_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32)] * 3
double_signature = single_signature + aspect_signature

if args.dataset == "res15" or args.dataset == "res16":
    num_asp_senti_pairs = len(RES1516_LABEL_MAPPING['texts']) * 3
elif args.dataset == "laptop_acos":
    num_asp_senti_pairs = len(ACOS_LAPTOP_LABEL_MAPPING['texts']) * 3
else:
    num_asp_senti_pairs = len(PHONE_CHINESE_LABEL_MAPPING['texts']) * 3

end_to_end_signature = [
                           tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                           tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                           tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                           tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                           tf.TensorSpec(shape=(None, None), dtype=tf.int32)
                       ] + aspect_signature

pretrain_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
] + aspect_signature


class ABSATrainer(object):

    def __init__(self, args, train_data, test_data, logger, checkpoint=None, cased=True, model_type="single_tower",
                 mask_sb=False, config=None, learning_rate=1e-5, dropout_rate=0.1, lang="en", drop_null_data=False):
        self.args = args
        self.logger = logger
        self.language = lang
        self.sentence_b = config['sentence_b']
        if config is None and checkpoint is None:
            raise ValueError("config and checkpoint must not be none at the same time!")
        self.model, self.optimizer, self.metrics, datasets, self.tokenizer, self.model_checkpoint = \
            prepare_modules(
                config=config,
                args=args,
                learning_rate=learning_rate,
                model_checkpoint=checkpoint,
                train_data_path=train_data,
                test_data_path=test_data,
                model_type=model_type,
                logger=self.logger,
                mask_sb=mask_sb,
                lang=lang,
                drop_null_data=drop_null_data
            )
        self.logger.info(self.model.get_config())
        self.train_dataset, self.test_dataset = datasets
        self.logger.info("Number of Train Sentence: {}".format(len(self.train_dataset)))
        self.model_type = model_type
        if self.model_type in ["single_tower", "double_tower"]:
            self.loss_metrics_names = ["Ner", "Target Aspect", "Target Sentiment"]
            self.acc_metrics_names = ["Ner", "Target Aspect", "Target Sentiment"]
        else:
            self.loss_metrics_names = ["Ner", "CLS Classification"]
            self.acc_metrics_names = ["Ner", "CLS Classification", "CLS Pos", "CLS Neg"]
        self.loss_metrics = [tf.keras.metrics.Mean(name=name + " " + "Loss") for name in self.loss_metrics_names]
        self.acc_metrics = [tf.keras.metrics.Mean(name=name + " " + "Acc") for name in self.acc_metrics_names]
        self.Result_tuple = namedtuple("result", ["target", "category", "polarity"])
        self.aspect_senti_batch_size = args.aspect_senti_batch_size
        self.aspect_senti_test_batch_size = args.aspect_senti_test_batch_size
        self.accumulated_gradients = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.model.trainable_variables]
        self.d_model = self.model.bert.config.hidden_size

    @tf.function(input_signature=[end_to_end_signature])
    def step(self, inputs):
        n_asp_senti = tf.shape(inputs[3])[1]
        n_asp_senti_batches = tf.constant(1) if self.aspect_senti_batch_size == -1 else tf.cast(tf.math.ceil(n_asp_senti / self.aspect_senti_batch_size), tf.int32)
        asp_senti_batch_size = n_asp_senti if self.aspect_senti_batch_size == -1 else tf.constant(self.aspect_senti_batch_size)
        text_states = tf.zeros((tf.shape(inputs[0])[0], tf.shape(inputs[0])[1], self.d_model))
        losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        accs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(n_asp_senti_batches):
            # tf.print("Aspect-Senti Bacht: ", i)
            start = i * asp_senti_batch_size
            end = (i + 1) * asp_senti_batch_size
            with tf.GradientTape() as tape:
                if isinstance(self.model, AspectSentimentModel):
                    loss, acc = self.model(
                        inputs=inputs,
                        phase="train"
                    )
                elif self.model_type == "double_tower":
                    texts_inputs = inputs[:-3]
                    aspect_inputs = inputs[-3:]
                    loss, acc = self.model(
                        text_inputs=texts_inputs,
                        aspect_inputs=aspect_inputs
                    )
                else:
                    text_inputs = inputs[:3]
                    label_inputs = [each[:, start: end] for each in inputs[3:5]]
                    aspect_senti_inputs = [each[start: end] for each in inputs[5:]]
                    loss, acc, _ = self.model(
                        text_inputs=text_inputs,
                        aspect_inputs=aspect_senti_inputs,
                        label_inputs=label_inputs,
                        asp_senti_batch_idx=start,
                        phase="train",
                        cache_text_states=text_states
                    )
            gradients = tape.gradient(loss[-1], self.model.trainable_variables)
            # tf.print("Loss: ", loss)
            # tf.print("Acc: ", acc)
            losses = losses.write(i, loss)
            accs = accs.write(i, acc)
            # accumulate grades
            for j, grad in enumerate(gradients):
                self.accumulated_gradients[j].assign_add(grad)
        self.optimizer.apply_gradients(zip(self.accumulated_gradients, self.model.trainable_variables))

        # clear grads
        for var in self.accumulated_gradients:
            var.assign(tf.zeros_like(var))

        loss = tf.reduce_mean(losses.stack(), axis=0)
        acc = tf.reduce_mean(accs.stack(), axis=0)
        self.metrics.update_state(loss[-1])
        for idx, loss_metric in enumerate(self.loss_metrics):
            loss_metric.update_state(loss[idx])
        for idx, acc_metric in enumerate(self.acc_metrics):
            acc_metric.update_state(acc[idx])

        return loss, acc


    @tf.function(input_signature=[pretrain_signature]) 
    def pretrain_step(self, inputs):
        with tf.GradientTape() as tape:
            text_inputs = inputs[:3]
            label_inputs = inputs[3:4]
            topic_inputs = inputs[4:]
            outputs = self.model(
                text_inputs=text_inputs,
                aspect_inputs=topic_inputs,
                label_inputs=label_inputs,
                phase="pretrain"
            )
        cls_loss = outputs["cls_loss"]
        cls_acc = outputs["cls_acc"]
        gradients = tape.gradient(cls_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.loss_metrics[-1].update_state(cls_loss)
        self.acc_metrics[-1].update_state(cls_acc) 
        return cls_loss, cls_acc


    def pretrain(self, data_loader, train_steps, save_steps=2000, log_steps=500, save_dir=None):
        loss_list = []
        acc_list = []
        iterator = iter(data_loader)
        num_steps = 0
        checkpoint_manager = tf.train.CheckpointManager(self.model_checkpoint, directory=save_dir, max_to_keep=3) if save_dir else None        
        while True:
            inputs = iterator.get_next()
            pretrain_loss, pretrain_acc = self.pretrain_step(inputs)
            loss_list.append(pretrain_loss.numpy())
            acc_list.append(pretrain_acc.numpy())
            num_steps += 1
            if num_steps % log_steps == 0:
                train_log = "Steps: {} || Loss: {:.4f} || Acc: {:.4f}".format(num_steps, self.loss_metrics[-1].result(), self.acc_metrics[-1].result())
                self.logger.info(train_log)
                self.loss_metrics[-1].reset_states()
                self.acc_metrics[-1].reset_states()
            if num_steps % save_steps == 0 and save_dir:
                self.logger.info("saving checkpoint...")
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                checkpoint_manager.save(num_steps)
                config = self.model.get_config()
                json.dump(config, open(os.path.join(save_dir, "model_config.json"), "w"))
            
            if num_steps >= train_steps:
                break
        
        self.logger.info("Pretraining Finish!")
        return loss_list, acc_list


    def train_and_eval(self, epoch, train_loader, valid_loader=None, valid_freq=1, test_freq=1, save_dir=None, do_train=True, do_valid=True, do_test=True):
        loss_list = []
        acc_list = []
        best_f1 = 0.0
        best_p = 0.0
        best_r = 0.0
        best_epoch = 0
        checkpoint_manager = tf.train.CheckpointManager(self.model_checkpoint, directory=save_dir if save_dir else "",
                                                        max_to_keep=1)
        for epoch in range(1, epoch + 1):
            n_steps = int(np.ceil(len(trainer.train_dataset) / self.args.train_batch_size))
            # train
            # for idx, inputs in tqdm(enumerate(train_loader), total=n_steps, desc=f"epoch {epoch}"):
            if do_train:
                for idx, inputs in enumerate(train_loader):
                    # print("Batch {}".format(idx))

                    # *************for speed_test*************
                    if args.speed_test:
                        global train_time
                        if idx == 1:
                            start_time = time.time()
                        elif idx == 21:
                            end_time = time.time()
                            train_time = (end_time - start_time) * n_steps / 20
                            break

                    train_loss, train_acc = self.step(inputs)
                    loss_list.append([item.numpy() for item in train_loss])
                    acc_list.append([item.numpy() for item in train_acc])

                # self.logger.info("Epoch {}: ".format(epoch))
                # self.logger.info("Train:")
                # self.logger.info(
                #     "Loss {:.4f}, NER Loss {:.4f}, Target Aspect Loss {:.4f}, Target Sentiment Loss {:.4f}".format(
                #         self.metrics.result(),
                #         self.loss_metrics[0].result(),
                #         self.loss_metrics[1].result(),
                #         self.loss_metrics[2].result(),
                #     ))
                loss_result = [(metric.name, metric.result().numpy().item()) for metric in self.loss_metrics]
                acc_result = [(metric.name, metric.result().numpy().item()) for metric in self.acc_metrics]
                loss_str = ",  ".join(str(each) for each in loss_result)
                acc_str = ",  ".join(str(each) for each in acc_result)
                # self.logger.info(
                #     "Loss {:.4f}, {}".format(
                #         self.metrics.result(),
                #         loss_str
                #     ))
                # self.logger.info("Ner Acc {:.4f}, Target Aspect Acc {:.4f}, Target Sentiment Acc {:.4f}".format(
                #     self.acc_mertrics[0].result(),
                #     self.acc_mertrics[1].result(),
                #     self.acc_mertrics[2].result()
                # ))
                # self.logger.info("{}".format(
                #     acc_str
                # ))
                train_log = "Epoch {} Train: Loss {:.4f}, {}, {}".format(epoch, self.metrics.result(), loss_str, acc_str)
                self.logger.info(train_log)   

                self.metrics.reset_states()
                for lm, am in zip(self.loss_metrics, self.acc_metrics):
                    lm.reset_states()
                    am.reset_states()

            # valid
            if do_valid and valid_loader and epoch % valid_freq == 0:
                valid_loss, valid_acc = self.evaluate(valid_loader)
                # self.logger.info("Evaluate:")
                valid_loss_result = [str((self.loss_metrics[i].name, valid_loss[i + 1])) for i in
                                     range(len(self.loss_metrics))]
                valid_acc_result = [str((self.acc_metrics[i].name, valid_acc[i])) for i in
                                    range(len(self.acc_metrics))]
                valid_loss_str = "Loss {:.4f}, {}".format(valid_loss[0], ",  ".join(valid_loss_result))
                valid_acc_str = ",  ".join(valid_acc_result)
                # self.logger.info(valid_loss_str)
                # self.logger.info(valid_acc_str)
                valid_log = "Epoch {} Valid: {}, {}".format(epoch, valid_loss_str, valid_acc_str)
                self.logger.info(valid_log)
            if do_test and epoch % test_freq == 0:
                t, p, l = self.test(self.test_dataset, self.args.test_batch_size)
                output_file_path = os.path.join(self.args.output_dir, self.args.task_name, f"epoch{epoch}_{args.dataset}.json") if self.args.output_dir else None
                # save test result
                if output_file_path is not None:
                    save_predict(t, p, l, output_file_path)

                precision, recall, f1 = compute_f1(p, l)
                self.logger.info("Epoch %d Test: Precision %.3f Recall %.3f F1 %.3f" % (epoch, precision, recall, f1))
                if f1 >= best_f1:
                    best_f1 = f1
                    best_p = precision
                    best_r = recall
                    best_epoch = epoch
                    if save_dir:
                        self.logger.info("saving best checkpoint...")
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        checkpoint_manager.save(epoch)
                        config = self.model.get_config()
                        json.dump(config, open(os.path.join(save_dir, "model_config.json"), "w"))
                # save
        # if save_dir:
        #     if not os.path.exists(save_dir):
        #         os.mkdir(save_dir)
        #     self.model_checkpoint.save(os.path.join(save_dir, "train"))
        #     config = self.model.get_config()
        #     json.dump(config, open(os.path.join(save_dir, "model_config.json"), "w"))
        self.logger.info(
            "Best Epoch {}; Best f1 {} precision {} recall {};".format(best_epoch, best_f1, best_p, best_r))
        return loss_list, acc_list

    @tf.function(input_signature=[end_to_end_signature])
    def evaluate_step(self, inputs):
        n_asp_senti = tf.shape(inputs[3])[1]
        n_asp_senti_batches = tf.constant(1) if self.aspect_senti_test_batch_size == -1 else tf.cast(tf.math.ceil(n_asp_senti / self.aspect_senti_test_batch_size), tf.int32)
        asp_senti_batch_size = n_asp_senti if self.aspect_senti_test_batch_size == -1 else tf.constant(self.aspect_senti_test_batch_size)
        losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        accs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        text_states = tf.zeros((tf.shape(inputs[0])[0], tf.shape(inputs[0])[1], self.d_model))
        for i in tf.range(n_asp_senti_batches):
            start = i * asp_senti_batch_size
            end = (i + 1) * asp_senti_batch_size
            if self.model_type == "single_tower":
                loss, acc = self.model(
                    inputs=inputs,
                    phase="valid"
                )
            elif self.model_type == "double_tower":
                texts_inputs = inputs[:-3]
                aspect_inputs = inputs[-3:]
                loss, acc = self.model(
                    text_inputs=texts_inputs,
                    aspect_inputs=aspect_inputs,
                    phase="valid"
                )
            else:
                texts_inputs = inputs[:3]
                label_inputs = [each[:, start: end] for each in inputs[3:5]]
                aspect_inputs = [each[start: end] for each in inputs[5:]]
                loss, acc, text_states = self.model(
                    text_inputs=texts_inputs,
                    aspect_inputs=aspect_inputs,
                    label_inputs=label_inputs,
                    cache_text_states=text_states,
                    asp_senti_batch_idx=start,
                    phase="valid"
                )
            losses = losses.write(i, loss)
            accs = accs.write(i, acc)
        
        loss = tf.reduce_mean(losses.stack(), axis=0)
        acc = tf.reduce_mean(accs.stack(), axis=0)

        
        return loss, acc

    @tf.function(input_signature=[end_to_end_signature[:3] + end_to_end_signature[-3:]])
    def test_step(self, inputs):
        n_asp_senti = tf.shape(inputs[3])[0]
        n_asp_senti_batches = tf.constant(1) if self.aspect_senti_test_batch_size == -1 else tf.cast(tf.math.ceil(n_asp_senti / self.aspect_senti_test_batch_size), tf.int32)
        asp_senti_batch_size = n_asp_senti if self.aspect_senti_test_batch_size == -1 else tf.constant(self.aspect_senti_test_batch_size)
        all_decoded_sequence = tf.TensorArray(dtype=tf.int32, size=n_asp_senti_batches, infer_shape=False)
        all_output_logits = tf.TensorArray(dtype=tf.float32, size=n_asp_senti_batches, infer_shape=False)
        all_output_cls_states = tf.TensorArray(dtype=tf.float32, size=n_asp_senti_batches, infer_shape=False)
        text_states = tf.zeros((tf.shape(inputs[0])[0], tf.shape(inputs[0])[1], self.d_model))
        for i in tf.range(n_asp_senti_batches):
            start = i * asp_senti_batch_size
            end = (i + 1) * asp_senti_batch_size
            if self.model_type == "single_tower":
                model_out = self.model(inputs, phase="test", output_attentions=False)
            elif self.model_type == "double_tower":
                texts_inputs = inputs[:-3]
                aspect_inputs = inputs[-3:]
                model_out = self.model(texts_inputs, aspect_inputs, phase="test", output_attentions=False)
            else:
                texts_inputs = inputs[:3]
                aspect_inputs = [each[start: end] for each in inputs[3:]]
                model_out = self.model(
                    texts_inputs, 
                    aspect_inputs,
                    cache_text_states=text_states,  
                    phase="test", 
                    asp_senti_batch_idx=start,
                    output_attentions=False
                )
            all_decoded_sequence = all_decoded_sequence.write(i, tf.transpose(model_out['decoded_sequence'], [1, 0, 2]))
            all_output_logits = all_output_logits.write(i, tf.transpose(model_out['output_logits'], [1, 0, 2, 3]))
            all_output_cls_states = all_output_cls_states.write(i, tf.transpose(model_out['output_cls_states'], [1, 0]))
            text_states = model_out['text_states']
        result = {
            "decoded_sequence": tf.transpose(all_decoded_sequence.concat(), [1, 0, 2]),
            "output_logits": tf.transpose(all_output_logits.concat(), [1, 0, 2, 3]),
            "output_cls_states": tf.transpose(all_output_cls_states.concat(), [1, 0]),
        }
        return result

    def evaluate(self, validation_loader):
        self.model.updated.assign(False)
        loss_list = []
        acc_list = []
        n_steps = int(np.ceil(len(self.test_dataset) / self.args.test_batch_size))
        for idx, inputs in enumerate(validation_loader):
            if args.speed_test:
                global eval_time
                if idx == 1:
                    start_time = time.time()
                elif idx == 11:
                    end_time = time.time()
                    eval_time = (end_time - start_time) * n_steps / 20
                    break            
            loss, acc = self.evaluate_step(inputs)
            loss_list.append([item.numpy() for item in loss])
            acc_list.append([item.numpy() for item in acc])
        total_loss = np.mean([each[-1] for each in loss_list])
        # ner_loss = np.mean([each[0] for each in loss_list])
        # clf_loss = np.mean([each[1] for each in loss_list])
        # sc_loss = np.mean([each[2] for each in loss_list])
        sub_losses = []
        sub_accs = []
        for i in range(len(self.loss_metrics_names)):
            sub_losses.append(np.mean([each[i] for each in loss_list]))
        
        for i in range(len(self.acc_metrics_names)):
            sub_accs.append(np.mean([each[i] for each in acc_list]))

        return [total_loss] + sub_losses, sub_accs

    def test(self, test_dataset, batch_size):
        test_string_loader = tf.data.Dataset.from_generator(
            test_dataset.generate_string_sample, output_types=(tf.string, tf.string)
        ).batch(batch_size)
        test_data_loader = test_string_loader.map(test_dataset.wrap_map)

        contexts = []
        parse_result = []
        true_result = []
        num_batch = len(test_dataset) // batch_size + 1
        for idx, (raw_strings, inputs) in enumerate(zip(test_string_loader, test_data_loader)):
            text, triplets = raw_strings[0].numpy(), raw_strings[1].numpy()
            text = [each.decode('utf-8') for each in text]
            contexts.extend(text)
            triplets = [each.decode('utf-8') for each in triplets]
            for t in triplets:
                one_label = []
                j_list = json.loads(t)
                for j in j_list:
                    one_label.append(Triplet(j['target'], j['category'], j['polarity']))
                true_result.append(set(one_label))

            if self.model_type == "double_tower":
                x_inputs = inputs[:4] + inputs[-3:]
                labels = inputs[4:-3]
            elif self.model_type == "single_tower":
                x_inputs = inputs[:4]
                labels = inputs[4:]
            else:
                x_inputs = inputs[:3] + inputs[5:]
                labels = inputs[3:5]
            model_out = self.test_step(x_inputs)

            # tokenized_texts = []
            # for i in range(len(text)):
            #     tokenized_text = self.tokenizer.tokenize(text[i], add_special_tokens=True)
            #     tokenized_texts.append(tokenized_text)

            tokenized_texts = self.tokenizer(text, padding='longest', return_offsets_mapping=True)
            batch_res = Result(self.tokenizer).get_result(model_out, tokenized_texts, text, label_mappings=self.sentence_b,
                                                          result_type=self.model_type, tagging_schema=self.args.schema)
            batch_res = [set(res) for res in batch_res]
            parse_result.extend(batch_res)
        return contexts, parse_result, true_result

def calculate_dataset_label_prior(datasets):
    f_p = 0
    f_all = 0
    for dataset in datasets:
        if dataset is not None:
            ds = tf.data.Dataset.from_generator(
                dataset.generate_string_sample,
                output_types=(tf.string, tf.string)
            )
            for a, b in ds.batch(32):
                cls_labels = dataset.map_batch_string_to_tensor_end_to_end(a, b)[4]
                f_p += cls_labels.sum()
                f_all += cls_labels.size
    p = f_p / f_all
    return np.array([1-p, p])


def prepare_modules(
        config,
        args,
        learning_rate=1e-5,
        model_checkpoint=None,
        train_data_path='data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml',
        test_data_path=None,
        model_type="single_tower",
        logger=logging.getLogger(),
        mask_sb=False,
        lang="en",
        drop_null_data=False,
):
    if model_checkpoint is not None:
        if os.path.isdir(model_checkpoint):
            ckpt_path = tf.train.latest_checkpoint(model_checkpoint)
            config_path = os.path.join(model_checkpoint, "model_config.json")
        else:
            ckpt_path = model_checkpoint
            dir_name = os.path.dirname(ckpt_path)
            config_path = os.path.join(dir_name, "model_config.json")
        if args.load_config:
            config = json.load(open(config_path))

    init_dir = config['cache_dir']
    init_model = config['init_bert_model']

    sentence_b = config['sentence_b']
    tokenizer = AutoTokenizer.from_pretrained(init_model, cache_dir=init_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if args.pretrain:
        Dataset = PreTrainDataset
    elif lang == "en":
        if args.dataset == 'res16' or args.dataset == 'res15':
            Dataset = EnglishDataset
        elif args.dataset == 'laptop_acos':
            Dataset = ACOSDataset
        else:
            raise ValueError(f"Error dataset {lang}, should be ['res15', 'res16', 'laptop_acos']")
    elif lang == "cn":
        Dataset = ChineseDataset
    else:
        raise ValueError(f"Error language {lang}")

    
    if args.pretrain:
        train_dataset = Dataset(
            file_path=train_data_path,
            tokenizer = tokenizer
        )
    else:
        train_dataset = Dataset(
            file_path=train_data_path,
            tokenizer=tokenizer,
            sentence_b=sentence_b,
            mask_sb=mask_sb,
            tagging_schema=config["tagging_schema"],
            model_type=model_type,
            drop_null=drop_null_data,
            neg_sample=args.neg_sample,
            data_sample_ratio=args.data_sample_ratio
        )

    if test_data_path:
        test_dataset = Dataset(
            file_path=test_data_path,
            tokenizer=tokenizer,
            sentence_b=sentence_b,
            mask_sb=mask_sb,
            tagging_schema=config["tagging_schema"],
            model_type=model_type,
            drop_null=drop_null_data,
            neg_sample=args.neg_sample if args.speed_test else -1
        )
    else:
        test_dataset = None
    
    # logit-adjustment
    if args.logit_adjust or args.detect_loss == "pwm":
        prior = calculate_dataset_label_prior([train_dataset])
        config["detect_label_prior"] = prior
        logger.info("Dataset CLS label prior: {}".format(prior))
    if model_type == "single_tower":
        model_class = AspectSentimentModel
    elif model_type == "double_tower":
        model_class = DoubleTowerAspectSentimentModel
    else:
        model_class = End2EndAspectSentimentModel
    model = model_class.from_config(config)
    if args.pretrain:
        model.te_block.trainable = False
    # model.build_params(input_shape=(None, model.bert.config.hidden_size))
    ckpt = tf.train.Checkpoint(model=model)
    if model_checkpoint is not None:
        logger.info("Loading parameters from checkpoint: %s" % ckpt_path)
        ckpt.restore(ckpt_path)
    model.build_params()
    # print(model.contain_dense.trainable_variables)      
    if args.decay_steps > 0:
        lr = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=args.decay_steps, decay_rate=args.decay_rate, staircase=True)
    else:
        lr = learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    metrics = tf.keras.metrics.Mean(name='train_loss')

    return model, optimizer, metrics, [train_dataset, test_dataset], tokenizer, ckpt


if __name__ == '__main__':
    set_seed(1)
    language = args.language

    if args.model_type in ["single_tower", "double_tower"]:
        sentence_b = SENTENCE_B['cased']
    elif language == "en":
        if args.dataset == "res16" or args.dataset == "res15":
            sentence_b = RES1516_LABEL_MAPPING
        elif args.dataset == "laptop_acos":
            sentence_b = ACOS_LAPTOP_LABEL_MAPPING      
    else:
        sentence_b = PHONE_CHINESE_LABEL_MAPPING

    # settings
    if language == "en":
        if args.cased:
            init_bert_model = f'bert-{args.bert_size}-cased'
            cache_dir = f'bert_models/bert-{args.bert_size}-cased'
        else:
            init_bert_model = f'bert-{args.bert_size}-uncased'
            cache_dir = f'bert_models/bert-{args.bert_size}-uncased'
    else:
        init_bert_model = f'bert-{args.bert_size}-chinese'
        cache_dir = f'bert_models/bert-{args.bert_size}-chinese'
    config = {
        "init_bert_model": init_bert_model,
        "sentence_b": sentence_b,
        "num_sentiment_classes": 3,
        "subblock_hidden_size": args.d_block,
        "subblock_head_num": args.block_att_head_num,
        "block_output_activation": args.block_output_activation,
        "block_inter_activation": args.block_inter_activation,
        "cache_dir": cache_dir,
        "fuse_strategy": args.fuse_strategy,
        "pooling": args.pooling,
        "tagging_schema": args.schema,
        "extra_attention": args.extra_attention,
        "hot_attention": args.hot_attention,
        "dropout": args.dropout_rate,
        "detect_dropout": args.detect_dropout_rate,
        "loss_ratio": args.loss_ratio,
        "detect_loss": args.detect_loss,
        "do_logit_adjust": args.logit_adjust,
        "tau": args.tau
    }
    model_type = args.model_type
    mask_sb = args.mask_sb
    learning_rate = args.lr

    checkpoint_path = args.init_model_dir if args.init_model_dir else None
    task_output_dir = os.path.join(args.output_dir, args.task_name)
    if args.output_dir.strip() and not os.path.exists(task_output_dir):
        print("Create output dir: {}".format(task_output_dir))
        os.makedirs(task_output_dir)
    log_path = os.path.join(task_output_dir,"train.log") if args.output_dir.strip() else None
    save_dir = os.path.join(args.save_dir, args.task_name)if args.save_dir.strip() else None
    # save_dir = "./checkpoint/end_2_end"
    # checkpoint_path = "./checkpoint/end_2_end"
    train_data_path, test_data_path = None, None
    if args.pretrain:
        train_data_path = f"./data/pretrain/{args.pretrain_data}"
    elif language == "en":
        if args.dataset == "res16":
            train_data_path = "./data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml"
            test_data_path = "./data/semeval2016/EN_REST_SB1_TEST_LABELED.xml"
        elif args.dataset == "res15":
            train_data_path = "./data/semeval2015/ABSA-15_Restaurants_Train_Final.xml"
            test_data_path = "./data/semeval2015/ABSA15_Restaurants_Test.xml"
        elif args.dataset == "laptop_acos":
            train_data_path = "./data/Laptop-ACOS/processed_data/laptop_quad_train.tsv"
            test_data_path = "./data/Laptop-ACOS/processed_data/laptop_quad_test.tsv"
    else:
        train_data_path = "./data/semeval2016/phone_chinese/labeled_phone.csv"
    # tokenizer = AutoTokenizer.from_pretrained(config['init_bert_model'], cache_dir=config['cache_dir'])
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # test_dataset = SemEvalDataSet(test_path, tokenizer, config['sentence_b'], mask_sb=mask_sb,
    #                               model_type=model_type)
    logger = logging.getLogger()
    prepare_logger(logger, level=logging.INFO, save_to_file=log_path)
    logger.info(args)

    trainer = ABSATrainer(
        args=args,
        train_data=train_data_path,
        test_data=test_data_path,
        logger=logger,
        checkpoint=checkpoint_path,
        cased=args.cased,
        model_type=model_type,
        mask_sb=mask_sb,
        config=config,
        learning_rate=learning_rate,
        lang=language,
        drop_null_data=args.drop_null_data
    )
    if args.data_aug is not None:
        logger.info("Data augment: Merge {} train data together".format(args.data_aug))
        trainer.train_dataset.augment(list(range(2, args.data_aug + 1)))
    
    if args.pretrain:
        train_loader = tf.data.Dataset.from_generator(
            trainer.train_dataset.generate_string_sample,
            output_types=(tf.string, tf.string, tf.int32)
        ).batch(batch_size=args.train_batch_size).map(trainer.train_dataset.wrap_map).repeat().prefetch(8)
    else:
        train_loader = tf.data.Dataset.from_generator(
            trainer.train_dataset.generate_string_sample,
            output_types=(tf.string, tf.string)
        ).batch(batch_size=args.train_batch_size)
        if args.cache_train_loader:
            train_loader = train_loader.map(trainer.train_dataset.wrap_map).cache().shuffle(buffer_size=10000).prefetch(8)
        else:
            train_loader = train_loader.shuffle(buffer_size=10000).map(trainer.train_dataset.wrap_map).prefetch(8)

    if args.do_test or args.do_valid:
        test_loader = tf.data.Dataset.from_generator(
            trainer.test_dataset.generate_string_sample,
            output_types=(tf.string, tf.string)
        ).batch(batch_size=args.test_batch_size).map(trainer.test_dataset.wrap_map).cache()
    else:
        test_loader = None
    

    if args.speed_test:
        speed_fp = 'train_eval_time.txt'
        if os.path.exists(speed_fp):
            speed_file = open(speed_fp, "a")
        else:
            speed_file = open(speed_fp, "w")
            speed_file.write(",".join(["method", "n_aspect", "train_time", "eval_time"]) + '\n')
        model_name = "TTEE"
        n_aspect = args.neg_sample // 3
        train_time = 0
        eval_time = 0

    if args.pretrain:
        train_loss, train_acc = trainer.pretrain(
            data_loader=train_loader,
            train_steps=args.pretrain_steps,
            save_steps=args.pretrain_save_steps,
            log_steps=args.pretrain_log_steps,
            save_dir=save_dir
        )
        import pandas as pd
        train_metric_df = pd.DataFrame({"train_loss": train_loss, "train_acc":train_acc})
        train_metric_df.to_csv(os.path.join(task_output_dir, "pretrain_metrics.csv"), index=False)

        
    else:
        train_loss, train_acc = trainer.train_and_eval(
            args.epochs,
            train_loader,
            test_loader,
            valid_freq=args.valid_freq,
            test_freq=args.test_freq,
            save_dir=save_dir,
            do_train=args.do_train,
            do_valid=args.do_valid,
            do_test=args.do_test
        )
    

    if args.speed_test:
        speed_file.write(",".join([model_name, str(n_aspect), str(train_time), str(eval_time)]) + "\n")
        speed_file.flush()
        speed_file.close()
    # p, l = trainer.test(trainer.test_dataset, args.test_batch_size)
    # precision, recall, f1 = compute_f1(p, l)
    # logger.info("Precision %.3f Recall %.3f F1 %.3f" % (precision, recall, f1))
    # test_tokenizer = TestTokenizer(trainer.tokenizer, SENTENCE_B['uncased'])
    # test_data = [
    #     "The food is fantastic, and the waiting staff has been perfect every single time we've been there."
    # ]
    # res, out = quick_test(test_data, test_tokenizer, trainer)
    # print(res)
