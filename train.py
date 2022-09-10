# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import logging
import argparse
import json
import random
import numpy as np
from models.single_tower_model import AspectSentimentModel
from models.double_tower_model import DoubleTowerAspectSentimentModel
from models.end_to_end_model import End2EndAspectSentimentModel
from datasets import ChineseDataset, EnglishDataset, TestTokenizer
from label_mappings import *
from transformers import AutoTokenizer
from utils.data_utils import prepare_logger
from utils.eval_utils import Result, Triplet, compute_f1
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
    parser.add_argument("--init_model_dir", help="热启动模型路径", default="", type=str)
    parser.add_argument("--save_dir", help="模型保存路径", default="", type=str)
    parser.add_argument("--log_path", help="日志保存路径", default="", type=str)
    parser.add_argument("--data_aug", help="数据增强策略", default=None, type=int)
    parser.add_argument("--dataset", help="数据集", default="res16", type=str)
    parser.add_argument("--train_batch_size", help="训练batch size", default=16, type=int)
    parser.add_argument("--test_batch_size", help="测试batch size", default=32, type=int)
    parser.add_argument("--epochs", help="训练epoch数量", default=30, type=int)
    parser.add_argument("--valid_freq", help="验证的频率", default=1, type=int)
    parser.add_argument("--lr", help="学习率", default=1e-5, type=float)
    parser.add_argument("--dropout_rate", help="失活率", default=0.1, type=float)
    parser.add_argument("--block_att_head_num", help="子模块自注意力头数", default=1, type=int)
    parser.add_argument("--fuse_strategy", help="端到端的FuseNet融合策略", default="concat", type=str)
    parser.add_argument("--extra_attention", help="端到端FuseNet之后是否加self Attention", default=False, action="store_true")
    parser.add_argument("--d_block", help="子模块模型维度", default=256, type=int)
    parser.add_argument("--model_type", help="模型种类（单塔、双塔、端到端）", default="end_to_end", type=str)
    parser.add_argument("--mask_sb", help="单塔模型attention mask, 不看sentence b", default=False, action="store_true")
    parser.add_argument("--cased", help="模型是否区分大小写", default=True, action="store_true")
    parser.add_argument("--do_train", help="是否进行训练", default=False, action="store_true")
    parser.add_argument("--do_valid", help="是否进行验证", default=False, action="store_true")
    parser.add_argument("--do_test", help="是否进行测试", default=False, action="store_true")
    parser.add_argument("--language", help="语言", default="en", type=str)
    parser.add_argument("--drop_null_data", help="是否在训练与测试集中去掉不包含三元组的句子", default=False, action="store_true")
    parser.add_argument("--loss_ratio", help="ner loss与detection loss的比例", default=1, type=float)
    return parser.parse_args()


language = "en"

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

if language == "en":
    num_asp_senti_pairs = len(ASPECT_SENTENCE['texts']) * 3
else:
    num_asp_senti_pairs = len(ASPECT_SENTENCE_CHINESE['texts']) * 3

end_to_end_signature = [
                           tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                           tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                           tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                           tf.TensorSpec(shape=(None, num_asp_senti_pairs, None), dtype=tf.int32),
                           tf.TensorSpec(shape=(None, num_asp_senti_pairs), dtype=tf.int32)
                       ] + aspect_signature


class ABSATrainer(object):

    def __init__(self, args, train_data, test_data, logger, checkpoint=None, cased=True, model_type="single_tower",
                 mask_sb=False, config=None, learning_rate=1e-5, dropout_rate=0.1, lang="en", drop_null_data=False):
        self.args = args
        self.logger = logger
        self.language = lang
        if config is None and checkpoint is None:
            raise ValueError("config and checkpoint must not be none at the same time!")
        self.model, self.optimizer, self.metrics, datasets, self.tokenizer, self.model_checkpoint = \
            prepare_modules(
                config=config,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                model_checkpoint=checkpoint,
                train_data_path=train_data,
                test_data_path=test_data,
                cased=cased,
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
            self.acc_metrics_names = ["Ner", "CLS Classification"]
        self.loss_metrics = [tf.keras.metrics.Mean(name=name + " " + "Loss") for name in self.loss_metrics_names]
        self.acc_mertrics = [tf.keras.metrics.Mean(name=name + " " + "Acc") for name in self.acc_metrics_names]
        self.Result_tuple = namedtuple("result", ["target", "category", "polarity"])

    @tf.function(input_signature=[end_to_end_signature])
    def step(self, inputs):
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
            elif self.model_type == "end_to_end":
                text_inputs = inputs[:3]
                label_inputs = inputs[3:5]
                aspect_senti_inputs = inputs[5:]
                loss, acc = self.model(
                    text_inputs=text_inputs,
                    aspect_inputs=aspect_senti_inputs,
                    label_inputs=label_inputs,
                    phase="train"
                )
        gradients = tape.gradient(loss[-1], self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.metrics.update_state(loss[-1])
        for idx, loss_metric in enumerate(self.loss_metrics):
            loss_metric.update_state(loss[idx])
        for idx, acc_metric in enumerate(self.acc_mertrics):
            acc_metric.update_state(acc[idx])

        return loss, acc

    def train_and_eval(self, epoch, train_loader, valid_loader=None, valid_freq=1, save_dir=None, do_test=True):
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
            for idx, inputs in tqdm(enumerate(train_loader), total=n_steps, desc=f"epoch {epoch}"):
                train_loss, train_acc = self.step(inputs)
                loss_list.append([item.numpy() for item in train_loss])
                acc_list.append([item.numpy() for item in train_acc])

            self.logger.info("Epoch {}: ".format(epoch))
            self.logger.info("Train:")
            # self.logger.info(
            #     "Loss {:.4f}, NER Loss {:.4f}, Target Aspect Loss {:.4f}, Target Sentiment Loss {:.4f}".format(
            #         self.metrics.result(),
            #         self.loss_metrics[0].result(),
            #         self.loss_metrics[1].result(),
            #         self.loss_metrics[2].result(),
            #     ))
            loss_result = [(metric.name, metric.result().numpy().item()) for metric in self.loss_metrics]
            acc_result = [(metric.name, metric.result().numpy().item()) for metric in self.acc_mertrics]
            loss_str = ",  ".join(str(each) for each in loss_result)
            acc_str = ",  ".join(str(each) for each in acc_result)
            self.logger.info(
                "Loss {:.4f}, {}".format(
                    self.metrics.result(),
                    loss_str
                ))
            # self.logger.info("Ner Acc {:.4f}, Target Aspect Acc {:.4f}, Target Sentiment Acc {:.4f}".format(
            #     self.acc_mertrics[0].result(),
            #     self.acc_mertrics[1].result(),
            #     self.acc_mertrics[2].result()
            # ))
            self.logger.info("{}".format(
                acc_str
            ))

            self.metrics.reset_states()
            for lm, am in zip(self.loss_metrics, self.acc_mertrics):
                lm.reset_states()
                am.reset_states()

            # valid
            if valid_loader and epoch % valid_freq == 0:
                valid_loss, valid_acc = self.evaluate(valid_loader)
                self.logger.info("Evaluate:")
                valid_loss_result = [str((self.loss_metrics[i].name, valid_loss[i + 1])) for i in
                                     range(len(self.loss_metrics))]
                valid_acc_result = [str((self.acc_mertrics[i].name, valid_acc[i])) for i in
                                    range(len(self.acc_mertrics))]
                valid_loss_str = "Loss {:.4f}, {}".format(valid_loss[0], ",  ".join(valid_loss_result))
                valid_acc_str = ", ".join(valid_acc_result)
                self.logger.info(valid_loss_str)
                self.logger.info(valid_acc_str)
                if do_test:
                    p, l = self.test(self.test_dataset, self.args.test_batch_size)
                    precision, recall, f1 = compute_f1(p, l)
                    self.logger.info("Precision %.3f Recall %.3f F1 %.3f" % (precision, recall, f1))
                    if f1 >= best_f1:
                        best_f1 = f1
                        best_p = precision
                        best_r = recall
                        best_epoch = epoch
                        if save_dir:
                            self.logger.info("saving best checkpoint...")
                            if not os.path.exists(save_dir):
                                os.mkdir(save_dir)
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
            aspect_inputs = inputs[5:]
            label_inputs = inputs[3:5]
            loss, acc = self.model(
                text_inputs=texts_inputs,
                aspect_inputs=aspect_inputs,
                label_inputs=label_inputs,
                phase="valid"
            )
        return loss, acc

    @tf.function(input_signature=[end_to_end_signature[:3] + end_to_end_signature[-3:]])
    def test_step(self, inputs):
        if self.model_type == "single_tower":
            model_out = self.model(inputs, phase="test", output_attentions=False)
        elif self.model_type == "double_tower":
            texts_inputs = inputs[:-3]
            aspect_inputs = inputs[-3:]
            model_out = self.model(texts_inputs, aspect_inputs, phase="test", output_attentions=False)
        else:
            texts_inputs = inputs[:3]
            aspect_inputs = inputs[3:]
            model_out = self.model(texts_inputs, aspect_inputs, phase="test", output_attentions=False)
        return model_out

    def evaluate(self, validation_loader):
        loss_list = []
        acc_list = []
        for idx, inputs in enumerate(validation_loader):
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
            sub_accs.append(np.mean([each[i] for each in acc_list]))

        return [total_loss] + sub_losses, sub_accs

    def test(self, test_dataset, batch_size):
        test_string_loader = tf.data.Dataset.from_generator(
            test_dataset.generate_string_sample, output_types=(tf.string, tf.string)
        ).batch(batch_size)
        test_data_loader = test_string_loader.map(test_dataset.wrap_map)

        parse_result = []
        true_result = []
        num_batch = len(test_dataset) // batch_size + 1
        for idx, (raw_strings, inputs) in tqdm(enumerate(zip(test_string_loader, test_data_loader)), total=num_batch):
            text, triplets = raw_strings[0].numpy(), raw_strings[1].numpy()
            text = [each.decode('utf-8') for each in text]
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
            batch_res = Result(self.tokenizer).get_result(model_out, tokenized_texts, text, language=self.language,
                                                          result_type=self.model_type)
            batch_res = [set(res) for res in batch_res]
            parse_result.extend(batch_res)
        return parse_result, true_result


def prepare_modules(
        config,
        learning_rate=1e-5,
        dropout_rate=0.1,
        model_checkpoint=None,
        train_data_path='data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml',
        test_data_path=None,
        cased=True,
        model_type="single_tower",
        logger=logging.getLogger(),
        mask_sb=False,
        lang="en",
        drop_null_data=False
):
    if model_checkpoint is not None:
        if os.path.isdir(model_checkpoint):
            ckpt_path = tf.train.latest_checkpoint(model_checkpoint)
            config_path = os.path.join(model_checkpoint, "model_config.json")
        else:
            ckpt_path = model_checkpoint
            dir_name = os.path.dirname(ckpt_path)
            config_path = os.path.join(dir_name, "model_config.json")
        config = json.load(open(config_path))

    init_dir = config['cache_dir']
    init_model = config['init_bert_model']

    sentence_b = config['sentence_b']
    tokenizer = AutoTokenizer.from_pretrained(init_model, cache_dir=init_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if lang == "en":
        Dataset = EnglishDataset
    elif lang == "cn":
        Dataset = ChineseDataset
    else:
        raise ValueError(f"Error language {lang}")

    train_dataset = Dataset(
        file_path=train_data_path,
        tokenizer=tokenizer,
        sentence_b=sentence_b,
        mask_sb=mask_sb,
        model_type=model_type,
        drop_null=drop_null_data
    )

    if test_data_path:
        test_dataset = Dataset(
            file_path=test_data_path,
            tokenizer=tokenizer,
            sentence_b=sentence_b,
            mask_sb=mask_sb,
            model_type=model_type,
            drop_null=drop_null_data
        )
    else:
        test_dataset = None
    if model_type == "single_tower":
        model_class = AspectSentimentModel
    elif model_type == "double_tower":
        model_class = DoubleTowerAspectSentimentModel
    else:
        model_class = End2EndAspectSentimentModel
    model = model_class.from_config(config)
    # model.build_params(input_shape=(None, model.bert.config.hidden_size))
    ckpt = tf.train.Checkpoint(model=model)
    if model_checkpoint is not None:
        logger.info("Loading parameters from checkpoint: %s" % ckpt_path)
        ckpt.restore(ckpt_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = tf.keras.metrics.Mean(name='train_loss')

    return model, optimizer, metrics, [train_dataset, test_dataset], tokenizer, ckpt


if __name__ == '__main__':
    set_seed(1)
    args = arg_parse()
    language = args.language

    if args.model_type in ["single_tower", "double_tower"]:
        sentence_b = SENTENCE_B['cased']
    elif language == "en":
        sentence_b = ASPECT_SENTENCE
    else:
        sentence_b = ASPECT_SENTENCE_CHINESE

    # settings
    if language == "en":
        if args.cased:
            init_bert_model = 'bert-base-cased'
            cache_dir = 'bert_models/bert-base-cased'
        else:
            init_bert_model = 'bert-base-uncased'
            cache_dir = 'bert_models/bert-base-uncased'
    else:
        init_bert_model = 'bert-base-chinese'
        cache_dir = 'bert_models/bert-base-chinese'
    config = {
        "init_bert_model": init_bert_model,
        "sentence_b": sentence_b,
        "num_sentiment_classes": 3,
        "subblock_hidden_size": args.d_block,
        "subblock_head_num": args.block_att_head_num,
        "cache_dir": cache_dir,
        "fuse_strategy": args.fuse_strategy,
        "dropout": args.dropout_rate,
        "extra_attention": args.extra_attention,
        "loss_ratio": args.loss_ratio
    }
    model_type = args.model_type
    mask_sb = args.mask_sb
    learning_rate = args.lr

    checkpoint_path = args.init_model_dir if args.init_model_dir else None
    log_path = args.log_path if args.log_path.strip() else None
    save_dir = args.save_dir if args.save_dir.strip() else None
    # save_dir = "./checkpoint/end_2_end"
    # checkpoint_path = "./checkpoint/end_2_end"
    train_data_path, test_data_path = None, None
    if language == "en":
        if args.dataset == "res16":
            train_data_path = "./data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml"
            test_data_path = "./data/semeval2016/EN_REST_SB1_TEST_LABELED.xml"
        elif args.dataset == "res15":
            train_data_path = "./data/semeval2015/ABSA-15_Restaurants_Train_Final.xml"
            test_data_path = "./data/semeval2015/ABSA15_Restaurants_Test.xml"
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
    train_loader = tf.data.Dataset.from_generator(
        trainer.train_dataset.generate_string_sample,
        output_types=(tf.string, tf.string)
    ).batch(batch_size=args.train_batch_size).shuffle(buffer_size=10000).map(trainer.train_dataset.wrap_map).prefetch(8)

    if args.do_test:
        test_loader = tf.data.Dataset.from_generator(
            trainer.test_dataset.generate_string_sample,
            output_types=(tf.string, tf.string)
        ).batch(batch_size=args.test_batch_size).map(trainer.test_dataset.wrap_map).cache()
    else:
        test_loader = None
    train_loss, train_acc = trainer.train_and_eval(
        args.epochs,
        train_loader,
        test_loader,
        valid_freq=args.valid_freq,
        save_dir=save_dir,
        do_test=args.do_test
    )

    # p, l = trainer.test(trainer.test_dataset, args.test_batch_size)
    # precision, recall, f1 = compute_f1(p, l)
    # logger.info("Precision %.3f Recall %.3f F1 %.3f" % (precision, recall, f1))
    # test_tokenizer = TestTokenizer(trainer.tokenizer, SENTENCE_B['uncased'])
    # test_data = [
    #     "The food is fantastic, and the waiting staff has been perfect every single time we've been there."
    # ]
    # res, out = quick_test(test_data, test_tokenizer, trainer)
    # print(res)
