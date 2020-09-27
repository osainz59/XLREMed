import os
import json
#import fire
from collections import defaultdict
from pprint import pprint
from itertools import product

from .dataset import Dataset


class DocRED(Dataset):

    def __init__(self, path):
        super(DocRED, self).__init__(name='DocRED')

        self.path = path

        self._init()
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _init(self):
        self.rel_info = json.load(open(os.path.join(self.path, 'rel_info.json')))
        self.rel2id = {v: i for i, v in enumerate(self.rel_info.keys())}
        self.train_path = os.path.join(self.path, 'train_annotated.json')
        self.train_dist_path = os.path.join(self.path, 'train_distant.json')
        self.dev_path = os.path.join(self.path, 'dev.json')
        self.test_path = os.path.join(self.path, 'test.json')

    def _read_instances(self, path, labels=False):
        with open(path, 'rt') as in_file:
            data = json.load(in_file)

        output = []
        for i, instance in enumerate(data):
            text = ""
            sentences_lenghts = []
            l = 0
            for sent in instance['sents']:
                sentences_lenghts.append(l)
                l += len(sent)
                text += " " + " ".join(sent)

            entities = []
            ent2id = defaultdict(list)
            for i, ent in enumerate(instance['vertexSet']):
                idx = f"#{i}"
                for elem in ent:
                    entities.append( (idx, elem['name'], sentences_lenghts[elem['sent_id']] + elem['pos'][0],
                                      sentences_lenghts[elem['sent_id']] + elem['pos'][1], elem['type']) )
                    ent2id[f"{elem['sent_id']}#{i}"].append(len(entities) - 1)

            if labels:
                relation_facts = []
                for label in instance['labels']:
                    heads, tails = [], []
                    for evidence in label['evidence']:
                        for h in ent2id.get(f"{evidence}#{label['h']}", []):
                            heads.append(h)
                        for t in ent2id.get(f"{evidence}#{label['t']}", []):
                            tails.append(t)
                    
                    for head, tail in product(heads, tails):
                        relation_facts.append( (self.rel2id[label['r']], head, tail) )
            text = self.tokenizer.encode(text)

            output.append( (text, entities) if not labels else (text, entities, relation_facts) )

        return output

    def get_train(self):
        if not self.train_data:
            self.train_data = self._read_instances(self.train_path, labels=True)
        return self.train_data

    def get_val(self):
        if not self.val_data:
            self.val_data = self._read_instances(self.dev_path, labels=True)
        return self.val_data

    def get_test(self):
        if not self.test_data:
            self.test_data = self._read_instances(self.test_path, labels=False)
        return self.test_data


def test():
    dataset = DocRED('data/DocRED')
    for instance in dataset.get_train():
        pprint(instance)
        break

if __name__ == "__main__":
    #fire.Fire(test)
    test()