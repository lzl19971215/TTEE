# encoding:utf-8
import requests
import requests
import random
import json
import time
from my_datasets import SemEvalDataSet
from transformers import AutoTokenizer
from label_mappings import *
from multiprocessing import Pool
from tqdm import tqdm

# client_id 为官网获取的AK， client_secret 为官网获取的SK



def post_translate_service(query):
    query = query  # example: hello
    # For list of language codes, please refer to `https://ai.baidu.com/ai-doc/MT/4kqryjku9#语种列表`
    from_lang = 'en'  # example: en
    to_lang = 'zh'  # example: zh
    term_ids = ''  # 术语库id，多个逗号隔开

    # Build request
    headers = {'Content-Type': 'application/json'}
    payload = {'q': query, 'from': from_lang, 'to': to_lang, 'termIds': term_ids}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    #     print(json.dumps(result, indent=4, ensure_ascii=False))
    trans_text = result['result']['trans_result'][0]['dst']
    return trans_text


def translate(data):
    text, triplets = data
    triplets = json.loads(triplets)
    count = 0
    for _ in range(5):
        try:
            trans_text = post_translate_service(text)
            break
        except:
            time.sleep(1)
    else:
        return None

    for t in triplets:
        if t["target"] != "NULL":
            for _ in range(5):
                try:
                    trans_target = post_translate_service(t["target"])
                    break
                except:
                    time.sleep(1)
            else:
                return None
            t["target"] = trans_target
    return trans_text, triplets


if __name__ == '__main__':
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=iDzLIvxveu8q8q3BeNx1AzpO' \
           '&client_secret=EVFkk4WCCbGuHt6NBn8zeU2qhvYsp1ik'
    response = requests.get(host)
    token = response.json()['access_token']
    url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + token

    data_path = 'data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', cache_dir='../models/bert-base-cased/')
    sentence_b = ASPECT_SENTENCE
    mask_sb = True
    model_type = 'end_to_end'
    dataset = SemEvalDataSet(
        file_path=data_path,
        tokenizer=tokenizer,
        sentence_b=sentence_b,
        mask_sb=mask_sb,
        model_type=model_type
    )

    pool = Pool(4)
    translate_result = pool.map(translate, tqdm(dataset.string_sentences))
    json.dump(translate_result, open("./data/semeval2016/translate_data.json", "w"))
