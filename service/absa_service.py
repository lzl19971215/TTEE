import os
import json
import sys
import logging
import tensorflow as tf

sys.path.append(".")
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
from flask import Flask, request
from my_datasets import TestTokenizer
from transformers import AutoTokenizer
from label_mappings import *
from online_predictor import OnlinePredictor
from utils.data_utils import prepare_logger

app = Flask(__name__)


@app.route('/absa_service', methods=['POST'])
def predict():
    data = request.get_data()
    logger.info("Get data: %s" % data)
    try:
        text_data = json.loads(data)
    except json.JSONDecodeError as e:
        result = {"status": "failed", "error": str(e), "error_code": 1}
        logger.error("Error: {}".format(json.dumps(result)))
        return json.dumps(result, ensure_ascii=False)

    # if not all(isinstance(x, str) for x in text_data):
    #     result = {"status": "failed", "error": "input should be of List[str]", "error_code": 2}
    #     logger.error("Input Format Error: {}".format(json.dumps(result)))
    #     return json.dumps(result, ensure_ascii=False)
    # if not isinstance(text_data, dict) or any(key not in text_data for key in ["text", "cn"]):
    #     result = {"status": "failed", "error": "input should be a list ", "error_code": 2}
    #     logger.error("Input Format Error: {}".format(json.dumps(result)))
    #     return json.dumps(result, ensure_ascii=False)

    try:
        text = text_data["text"]
        language = text_data["language"]
        output_attentions = text_data["output_attentions"] if "output_attentions" in text_data else False
        if language == "cn":
            result = cn_online_predictor.predict(text, output_attentions=output_attentions)
        elif language == "en":
            result = en_online_predictor.predict(text, output_attentions=output_attentions)
        else:
            result = {"status": "failed", "error": "language should be in [cn, en]", "error_code": 2}
        result["error_code"] = 0
    except Exception as e:
        result = {"status": "failed", "error": str(e), "error_code": 3}
        logger.error("Inner Error: {}".format(json.dumps(result)))
        return json.dumps(result, ensure_ascii=False)
    logger.info("Success")
    return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    cn_saved_model_path = "saved_models/cn_aug23_cache"
    en_saved_model_path = "saved_models/en_aug23_cache"
    cn_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir="../models/bert-base-chinese")
    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir="../models/bert-base-cased")
    cn_test_tokenizer = TestTokenizer(cn_tokenizer, RES1516_LABEL_MAPPING, False, "end_to_end")
    en_test_tokenizer = TestTokenizer(en_tokenizer, PHONE_CHINESE_LABEL_MAPPING, False, "end_to_end")
    cn_online_predictor = OnlinePredictor(cn_saved_model_path, cn_test_tokenizer, "cn")
    en_online_predictor = OnlinePredictor(en_saved_model_path, en_test_tokenizer, "en")

    logger = logging.getLogger()
    prepare_logger(logger, level="INFO", save_to_file="service/service.log")
    app.run(host="0.0.0.0", port=3030, debug=False)


