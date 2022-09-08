# -*- coding: utf-8 -*-
import logging
from train import ABSATrainer
from label_mappings import *
from transformers import AutoTokenizer
from datasets import TestTokenizer
from utils.eval_utils import quick_test

if __name__ == '__main__':
    cased = True
    mask_sb = True
    model_type = "end_to_end"

    init_dir = '../models/bert-base-chinese'
    init_model = 'bert-base-chinese'
    data_path = 'data/semeval2016/phone_chinese/labeled_phone.csv'
    checkpoint_path = "./checkpoint/cn_e2e_30_2_layer_pool_256_h_aug23"
    lang = "cn"
    test_path = None

    tokenizer = AutoTokenizer.from_pretrained(init_model, cache_dir=init_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    trainer = ABSATrainer(
        None,
        data_path,
        test_path,
        logger=logging.getLogger(),
        checkpoint=checkpoint_path,
        cased=cased,
        model_type=model_type,
        mask_sb=mask_sb,
        lang=lang
    )
    test_tokenizer = TestTokenizer(trainer.tokenizer, ASPECT_SENTENCE_CHINESE, mask_sb, model_type)

    test_data = [
        "屏幕质量不错，很抗摔；电池续航很好，可以待机两天；相机不是很给力，拍出来有点模糊"
    ]

    res, _ = quick_test(test_data, test_tokenizer, trainer, output_attentions=True)
    print(res)
