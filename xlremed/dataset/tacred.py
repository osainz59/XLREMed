import os
import json
#import fire
from collections import defaultdict
from pprint import pprint
from itertools import product
import numpy as np
import torch
import random

from tqdm import tqdm

from .dataset import Dataset, BatchLoaderSentence, BatchLoaderEntityPair

def get_relation_information(path):
    try:
        import pandas as pd

        info = pd.read_csv(path, sep='\t', header=0)
        info.drop(len(info)-1, inplace=True)

        rel2id = {r: i for i, r in enumerate(info['# Relation'])}
        class_weights = np.array([int(rel) for rel in info['Train']])
        class_weights = list(class_weights.sum() / class_weights)
    except:
        import csv

        with open(path) as in_file:
            info = []
            rel2id = {}
            class_weights = []
            for i, row in enumerate(csv.reader(in_file, delimiter='\t')):
                if i == 0:
                    continue
                info.append(row)
                rel2id[row[0]] = i-1
                class_weights.append(int(row[3]))
            #del info[0]
            del rel2id['Total']
            del class_weights[-1] 

            class_weights = np.array(class_weights)
            class_weights = list(class_weights.sum() / class_weights)
    
    return info, rel2id, class_weights


class TACREDEntityPair(Dataset):
    """
    """

    def __init__(self, path, pretrained_model, skip_na=False):
        super(TACREDEntityPair, self).__init__(name='TACRED_ep', pretrained_model=pretrained_model)

        self.path = path
        self.skip = skip_na

        self._init()
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _init(self):
        self.train_path = os.path.join(self.path, 'data/json/train.json')
        self.dev_path = os.path.join(self.path, 'data/json/dev.json')
        self.test_path = os.path.join(self.path, 'data/json/test.json')

        # Relation information
        self.info, self.rel2id, self.class_weights = get_relation_information(os.path.join(self.path, 'docs/tacred_stats.tsv'))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

    def _read_instances(self, path, batch_size, labels=False):
        with open(path, 'rt') as in_file:
            data = json.load(in_file)

        # Group instances by documents
        instances = []
        progress = tqdm(enumerate(data), desc=f"Loading dataset: {path}", total=len(data))
        for i, inst in progress:
            instances.append(TACREDEntityPairInstance(inst, self, i))

        return BatchLoaderEntityPair(instances, batch_size, self.tokenizer.pad_token_id, labels=labels)

    def get_train(self, batch_size):
        if not self.train_data:
            self.train_data = self._read_instances(self.train_path, batch_size, labels=True)
        return self.train_data

    def get_val(self, batch_size):
        if not self.val_data:
            self.val_data = self._read_instances(self.dev_path, batch_size, labels=True)
        return self.val_data

    def get_test(self, batch_size):
        if not self.test_data:
            self.test_data = self._read_instances(self.test_path, batch_size, labels=True)
        return self.test_data


class TACREDEntityPairInstance(object):
    """
    """

    def __init__(self, instance, dataset, result_pos):
        super(TACREDEntityPairInstance, self).__init__()

        self.rel2id = dataset.rel2id
        self.docid = instance['docid']
        self.tokens = instance['token']
        self.pos = instance['stanford_pos']
        self.ner = instance['stanford_ner']
        self.deptree = instance['stanford_deprel']
        self.heads = instance['stanford_head']
        self.relation = dataset.rel2id[instance['relation']]
        self.result_pos = result_pos
        self.tokenizer = dataset.tokenizer

        new_tokens = [dataset.tokenizer.cls_token_id]
        ent_pos, ent_end_pos, pos_order = [], [], []
        pos = 1
        E1S_TOKEN = dataset.tokenizer.encode('[E1S]', add_special_tokens=False)
        E1E_TOKEN = dataset.tokenizer.encode('[E1E]', add_special_tokens=False)
        E2S_TOKEN = dataset.tokenizer.encode('[E2S]', add_special_tokens=False)
        E2E_TOKEN = dataset.tokenizer.encode('[E2E]', add_special_tokens=False)

        for i, token in enumerate(self.tokens):
            new_token = dataset.tokenizer.encode(token, add_special_tokens=False)
            if i in [instance['subj_start'], instance['obj_start']]:
                ent_pos.append(pos)
                pos_order.append( int(i == instance['obj_start']) ) # Check which argument comes first
                new_tokens.extend(E1S_TOKEN if i == instance['subj_start'] else E2S_TOKEN)
                pos += 1

            new_tokens.extend(new_token)
            pos += len(new_token)
            if i in [instance['subj_end'], instance['obj_end']]:
                ent_end_pos.append(pos)
                new_tokens.extend(E1E_TOKEN if i == instance['subj_end'] else E2E_TOKEN)
                pos += 1
        
        self.input_tokens = torch.tensor(new_tokens)
        self.ent_pos = ent_pos
        self.ent_end_pos = ent_end_pos
        
        # Swap the order of position to maintain the head-tail order
        if pos_order != [0, 1]:
            self.ent_pos[0], self.ent_pos[1] = self.ent_pos[1], self.ent_pos[0]
            self.ent_end_pos[0], self.ent_end_pos[1] = self.ent_end_pos[1], self.ent_end_pos[0]

    def get_masked_input(self, probability=.5):
        """ TODO: Finish and test
        """
        masked_input_tokens = self.input_tokens.clone()
        # Mask first entity
        if random.random() > probability:
            masked_input_tokens[self.ent_pos[0]+1:self.ent_end_pos[0]] = self.tokenizer.mask_token_id
        # Mask second entity
        if random.random() > probability:
            masked_input_tokens[self.ent_pos[1]+1:self.ent_end_pos[1]] = self.tokenizer.mask_token_id
        
        return masked_input_tokens

    def evaluate(self, outs):
        predictions = [np.argmax(outs)]
        labels = [self.relation]
        positions = [self.result_pos]

        return predictions, labels, positions


class TACREDSentence(Dataset):
    """ TACRED dataset class. 

    This class handles TACRED dataset. It converts the TACRED data in json format into
    a standard suitable input for the model.

    TODO: Implement skip_na option.

    Usage:
    ```
    >>> from graphre.dataset import TACRED
    
    >>> data = TACRED('path/to/dataset')
    >>> train_inst = data.get_train(batch_size=4)
    ```
    """

    def __init__(self, path, pretrained_model, skip_na=False):
        super(TACREDSentence, self).__init__(name='TACRED', pretrained_model=pretrained_model)

        self.path = path
        self.skip = skip_na

        self._init()
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _init(self):
        self.train_path = os.path.join(self.path, 'data/json/train.json')
        self.dev_path = os.path.join(self.path, 'data/json/dev.json')
        self.test_path = os.path.join(self.path, 'data/json/test.json')

        # Relation information
        self.info, self.rel2id, self.class_weights = get_relation_information(os.path.join(self.path, 'docs/tacred_stats.tsv'))
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        # self.info = pd.read_csv(os.path.join(self.path, 'docs/tacred_stats.tsv'), sep='\t', header=0)
        # self.info.drop(len(self.info)-1, inplace=True)
        # No-relation index = 0
        # self.rel2id = {r: i for i, r in enumerate(self.info['# Relation'])}

    def _read_instances(self, path, batch_size, labels=False):
        with open(path, 'rt') as in_file:
            data = json.load(in_file)

        # Group instances by documents
        data_by_doc = dict()
        progress = tqdm(enumerate(data), desc=f"Loading dataset: {path}")
        for i, inst in progress:
            doc = str(inst['token'])
            if doc not in data_by_doc:
                data_by_doc[doc] = TACREDSentenceInstance(inst, self)
            data_by_doc[doc].add_relation(inst, i)

        return BatchLoaderSentence(list(data_by_doc.values()), batch_size, self.tokenizer.pad_token_id, labels=labels)

    def get_train(self, batch_size):
        if not self.train_data:
            self.train_data = self._read_instances(self.train_path, batch_size, labels=True)
        return self.train_data

    def get_val(self, batch_size):
        if not self.val_data:
            self.val_data = self._read_instances(self.dev_path, batch_size, labels=True)
        return self.val_data

    def get_test(self, batch_size):
        if not self.test_data:
            self.test_data = self._read_instances(self.test_path, batch_size, labels=False)
        return self.test_data


class TACREDSentenceInstance(object):

    def __init__(self, instance, dataset):
        super(TACREDSentenceInstance, self).__init__()

        self.rel2id = dataset.rel2id
        self.docid = instance['docid']
        self.tokens = instance['token']
        self.pos = instance['stanford_pos']
        self.ner = instance['stanford_ner']
        self.deptree = instance['stanford_deprel']
        self.heads = instance['stanford_head']

        self.relations = []
        self.entities = {}
        self.ent_pos = []

        self.input_tokens, self.token_positions = dataset.token2wordpiece(self.tokens)

    def add_relation(self, instance, result_pos):
        head = " ".join(self.tokens[instance['subj_start']:instance['subj_end']+1])
        tail = " ".join(self.tokens[instance['obj_start']:instance['obj_end']+1])
        if head not in self.entities:
            self.entities[head] = len(self.entities)
            self.ent_pos.append((self.token_positions[instance['subj_start']],
                                 self.token_positions[instance['subj_end']]))
        if tail not in self.entities:
            self.entities[tail] = len(self.entities)
            self.ent_pos.append((self.token_positions[instance['obj_start']],
                                 self.token_positions[instance['obj_end']]))

        self.relations.append(
            {
                'head': {
                    'idx': self.entities[head],
                    'start': self.token_positions[instance['subj_start']],
                    'end': self.token_positions[instance['subj_end']],
                    'type': instance['subj_type']
                },
                'tail': {
                    'idx': self.entities[tail],
                    'start': self.token_positions[instance['obj_start']],
                    'end': self.token_positions[instance['obj_end']],
                    'type': instance['obj_type']
                },
                'relation': instance['relation'],
                'result_pos': result_pos
            }
        )

    def get_entity_matrix(self, seq_len):
        adj = torch.zeros((seq_len, seq_len))
        for start, end in self.ent_pos:
            adj[start, start:end+1] = 1.
        
        return adj

    def get_label_matrix(self, seq_len):
        adj = torch.zeros((seq_len, seq_len)).long()
        for r in self.relations:
            adj[r['head']['start'], r['tail']['start']] = self.rel2id[r['relation']]

        return adj

    def evaluate(self, adj):
        predictions, labels, positions = [], [], []
        for relation in self.relations:
            max_r = np.argmax(adj[:, relation['head']['start'], relation['tail']['start']])
            predictions.append(max_r)
            labels.append(self.rel2id[relation['relation']])
            positions.append(relation['result_pos'])

        return predictions, labels, positions


    def __repr__(self):
        return "\n".join([self.docid, str(self.tokens), str(self.entities), str(self.relations)])


def test():
    tacred = TACREDEntityPair('data/tacred', 'bert-base-uncased')
    train = tacred.get_val(batch_size=5)
    print(len(train))
    seq, mask, ent, lab = train.get_batch(0, mask_p=.5)
    print(seq)
    print(mask)
    print(ent)
    print(lab)


if __name__ == "__main__":
    #fire.Fire(test)
    test()



