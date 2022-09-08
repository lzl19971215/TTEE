import tensorflow as tf
import time
from typing import List
from utils.eval_utils import Result


class OnlinePredictor(object):
    def __init__(self, saved_model_path, test_tokenizer, language):
        self.saved_model = tf.saved_model.load(saved_model_path)
        self.tokenizer = test_tokenizer
        self.language = language

    def predict(self, text_data: List[str], output_attentions=False):
        start = time.time()
        tokenized_data = self.tokenizer.tokenize(text_data, max_len=32)
        text_inputs = [tokenized_data['input_ids'], tokenized_data['token_type_ids'], tokenized_data['attention_mask']]
        aspect_inputs = [tokenized_data['aspect_input_ids'], tokenized_data['aspect_token_type_ids'],
                         tokenized_data['aspect_attention_mask']]
        output = self.saved_model.call(text_inputs, aspect_inputs, phase="test", output_attentions=output_attentions)
        result = Result(tokenizer=self.tokenizer.tokenizer)
        parse_result = result.get_result(model_output=output, tokenized_texts=tokenized_data, origin_texts=text_data,
                                         language=self.language, result_type="end_to_end")
        end = time.time()
        return_dict = {
            "origin_text": text_data,
            "result": eval(parse_result.__str__()),
            "time_cost": end - start,
            "status": "success"
        }
        if output_attentions:
            return_dict["attentions"] = output["sim_matrix"].numpy().tolist()
        return return_dict


