import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from utils.eval_utils import compute_f1


class Task(object):

    def __init__(self, task_name, task_elements) -> None:
        self.task_name = task_name
        self.task_elements = task_elements
        self.Result = namedtuple(task_name, task_elements)
        self.contexts = []
        self.predicts = []
        self.labels = []

    def _json_to_result(self, jobj):
        e = [jobj[k] for k in self.task_elements]
        return self.Result(*e)

    def parse(self, context, predict, label):
        p_set = set(self._json_to_result(each) for each in predict)
        l_set = set(self._json_to_result(each) for each in label)
        self.contexts.append(context)
        self.predicts.append(p_set)
        self.labels.append(l_set)

    def evaluate(self):
        return compute_f1(self.predicts, self.labels)

    def reset(self):
        self.contexts = []
        self.predicts = []
        self.labels = []


class MultiTaskEvaluatior(object):

    def __init__(self, task_config, output_dir) -> None:
        self.task_pools = {}
        self.output_dir = output_dir
        for name, elements in task_config.items():
            self.task_pools[name] = Task(name, elements)

        self.output_results = [json.load(open(os.path.join(
            output_dir, fn))) for fn in os.listdir(output_dir) if fn.endswith(".json")]
        self.eval_results = {k: {"p": [], "r": [], "f1": []}
                             for k in task_config}

    def eval_epoch_task(self, epoch, task_name):
        self.parse(epoch)
        self.eval(task_name)

    def parse(self, epoch):
        output_result = self.output_results[epoch]
        for test_case in output_result:
            context = test_case["context"]
            for task in self.task_pools.values():
                task.parse(context, test_case["predict"], test_case["label"])

    def eval(self, task_name):
        task = self.task_pools[task_name]
        p, r, f1 = task.evaluate()
        self.eval_results[task.task_name]["p"].append(p)
        self.eval_results[task.task_name]["r"].append(r)
        self.eval_results[task.task_name]["f1"].append(f1)

    def reset(self):
        for task in self.task_pools.values():
            task.reset()

    def display(self, save=False):
        best_results = []
        for name, results in self.eval_results.items():
            best_epoch = np.argmax(results["f1"])
            best_results.append({
                "epoch": best_epoch+1,
                "p": results["p"][best_epoch],
                "r": results["r"][best_epoch],
                "f1": results["f1"][best_epoch]
            })
        df = pd.DataFrame(data=best_results, index=self.eval_results.keys())
        if save:
            df.to_csv(os.path.join(self.output_dir, "eval_results.csv"))

        print(df)
        print()
        return df

    def full_evaluate(self, save=False):
        for epoch in tqdm(range(len(self.output_results))):
            self.parse(epoch)
            for task_name in self.task_pools:
                self.eval(task_name)
            self.reset()
        return self.display(save=save)


TASK_CONFIG = {
    "AS": ["aspect", "polarity"],
    "TS": ["target", "polarity"],
    "TA": ["target", "aspect"],
    "A": ["aspect"],
    "T": ["target"],
    "TAS": ["target", "aspect", "polarity"]
}

if __name__ == "__main__":
    res15_results = []
    res16_results = []
    for exp in os.listdir("output"):
        print(exp)
        if "eval_results.csv" in os.listdir(f"output/{exp}"):
            df = pd.read_csv(f"output/{exp}/eval_results.csv", index_col="Unnamed: 0")
            print(df)
            print()
            res15_results.append((exp,df)) if "res15" in exp else res16_results.append((exp,df))
            continue
        try:
            evaluator = MultiTaskEvaluatior(TASK_CONFIG, f"output/{exp}")
            df = evaluator.full_evaluate(True)
            res15_results.append((exp,df)) if "res15" in exp else res16_results.append((exp,df))
        except:
            print("exp result not complete!")
            print()
            continue
    


    def get_best_score_each_task(results):
        res = {"AS":[], "TS":[], "TA":[], "A":[], "T":[], "TAS":[]}
        for exp, df in results:
            for task in res:
                res[task].append((df.loc[task]["f1"], exp))
        for k,v in res.items():
            v.sort(key=lambda x:x[0], reverse=True)
            print(f"{k}: Best f1:{v[0][0]}; Experiment:{v[0][1]}")
        return res
    print("Res15:")
    res15_scores = get_best_score_each_task(res15_results)
    print("\nRes16:")
    res16_scores = get_best_score_each_task(res16_results)



        
