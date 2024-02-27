from collections import namedtuple
from transformers import BertTokenizer

def process_acos_data(raw_data_path, save_path, tokenizer: BertTokenizer):
    print(f"Start to proccess {raw_data_path}")
    categories = set()
    Triplet = namedtuple('triplet', ['target', 'category', 'polarity', 'start', 'end'])
    with open(raw_data_path, encoding='utf-8') as fr, open(save_path, encoding='utf-8', mode='w') as fw:
        for row in fr:
            row = row.strip().split('\t')
            text = row[0]
            quotes = row[1:]
            split_text = text.split()
            origin_text = tokenizer.convert_tokens_to_string(split_text)
            
            triplets = set()
            for quote in quotes:
                items = quote.split()
                start, end = items[0].split(',')
                target = None if start == end == '-1' else split_text[int(start): int(end)]
                if target:
                    target = tokenizer.convert_tokens_to_string(target)
                    newStart = origin_text.index(target)
                    newEnd = newStart + len(target)
                else:
                    newStart = 0
                    newEnd = 0
                category = items[1]
                categories.add(category)
                if items[2] == '0':
                    polarity = 'negative'
                elif items[2] == '1':
                    polarity = 'neutral'
                elif items[2] == '2':
                    polarity = 'positive'
                else:
                    raise ValueError(f'Unexpected sentiment code: {items[2]}')
                
                triplets.add(Triplet(target, category, polarity, newStart, newEnd))
            triplets_list = []
            for tup in triplets:
                triplets_list.append(" ".join([
                    f"{tup.start},{tup.end}",
                    tup.category,
                    tup.polarity
                ]))

            if len(triplets_list):
                trip = '\t'.join(triplets_list)
                res = f"{origin_text}\t{trip}\n"
                fw.write(res)
            else:
                print(f"No triplet find for {origin_text}")
    print(f"Finish proccessing{raw_data_path}, save to {save_path}")
    return categories


if __name__ == "__main__":
    import json
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='../../bert_models/bert-base-uncased')
    train_categories = process_acos_data('./raw_data/laptop_quad_train.tsv', './processed_data/laptop_quad_train.tsv', tokenizer)
    dev_categories = process_acos_data('./raw_data/laptop_quad_dev.tsv', './processed_data/laptop_quad_dev.tsv', tokenizer)
    test_categories = process_acos_data('./raw_data/laptop_quad_test.tsv', './processed_data/laptop_quad_test.tsv', tokenizer)
    all_cate = list(train_categories | dev_categories | test_categories)
    all_cate.sort(key=lambda x: x.split('#'))
    texts = [" of ".join(each.lower().replace("_", " ").split("#")[::-1]) for each in all_cate]
    laptop_mapping = { 
        'texts': texts, 
        'sentiments': [
            "negative",
            "neutral",
            "positive"
        ],
        'category2index': {c: idx for (idx, c) in enumerate(all_cate)}
    }
    json.dump(laptop_mapping, open('../../acos_laptop_mapping.json', 'w', encoding='utf-8'))
    print("Dump laptop aspect-sentiment mapping to ../../acos_laptop_mapping.json")