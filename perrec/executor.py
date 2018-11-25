# encoding=utf-8

import json

class BaseExecutor(object):
    def __init__(self, name, dataset, scoring):
        self.name = name
        self.dataset = dataset
        self.scoring = scoring

    def run(self):
        pass

    def get_result_file(self, data_dir):
        pass

    def dump_result(self, data_dir, scores):
        result_file = self.get_result_file(data_dir)
        for key, value in scores.items():
            scores[key] = list(value)
        with open(result_file, 'w') as f:
            json.dump(scores, f)
    
    def load_result(self, data_dir):
        result_file = self.get_result_file(data_dir)
        with open(result_file, 'r') as f:
            scores = json.load(f)
        return scores
