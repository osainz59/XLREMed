import os
import json
#import fire
from collections import defaultdict, Counter
from pprint import pprint
from itertools import product
import numpy as np
import torch
import random

from tqdm import tqdm

from .dataset import Dataset, BatchLoaderSentence, BatchLoaderEntityPair


class EHealthKD(Dataset):

    def __init__(self, path, pretrained_model, add_ensemble_data=False):
        super(EHealthKD, self).__init__(name='EHealthKD', pretrained_model=pretrained_model)

        self.path = path
        self._init()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.add_ensemble_data = add_ensemble_data

    def _init(self):
        self.train_path = os.path.join(self.path, 'data/training/')
        self.dev_path = os.path.join(self.path, 'data/development/main/')
        self.ensemble_path = os.path.join(self.path, 'data/ensemble/')
        self.test_path = os.path.join(self.path, 'data/testing/scenario3-taskB/')
        #self.test_path = os.path.join(self.path, 'data/json/test.json')

        self.id2ner = ['Other', 'Concept', 'Action', 'Predicate', 'Reference']
        self.ner2id = {name: idx for idx, name in enumerate(self.id2ner)}

        self.id2rel = ['no-relation', 'is-a', 'same-as', 'has-property', 'part-of', 'causes', 'entails',
                       'in-time', 'in-place', 'in-context', 'subject', 'target', 'domain',
                       'arg']
        self.rel2id = {name: idx for idx, name in enumerate(self.id2rel)}

    def _read_instances(self, path, batch_size, labels=False, instances_type='entity_pair', verbose=False):
        """ TODO

        instances_type in ['sentence', 'entity_pair']
        """
        with open(os.path.join(path, 'scenario.txt'), 'rb') as txtfile:
            instances, inst_len = [], []
            for line in txtfile:
                line = line.decode('utf-8').rstrip()
                instances.append({
                    'txt': line,
                    'terms': [],
                    'relations': []
                })
                prev = inst_len[-1] if len(inst_len) > 0 else 0
                inst_len.append(prev + len(line) + 1)       # the +1 for the \n

        inst_len = np.array(inst_len)
        #print(inst_len)
        def get_instance_id(char_id, char2_id=None):
            idx = int((inst_len <= char_id).sum())
            if char2_id is not None:
                idx2 = int((inst_len <= char_id).sum())
                return idx, idx2
            return idx

        terms, relations = {}, {}
        same_as_n = 0
        with open(os.path.join(path, 'scenario.ann'), 'rb') as annfile:
            for line in annfile:
                line = line.decode('utf-8').strip().split('\t')
                t = line[0]
                args = line[1:]
                
                if t.startswith('T'): #Term
                    text = args[-1]
                    label = args[0].split()[0]
                    positions = " ".join(args[0].split()[1:])
                    positions = [tuple(map(int, word.strip().split())) for word in positions.split(';')]
                    text_id = get_instance_id(positions[0][0], positions[-1][0])
                    
                    assert text_id[0] == text_id[1]
                    if text_id[0]:
                        rel_dist = inst_len[text_id[0] - 1]
                        rel_positions = [(pos_start - rel_dist, pos_end - rel_dist) for pos_start, pos_end in positions]

                    terms[t] = {
                        'text': text,
                        'ner-label': label,
                        'text_id': text_id[0],
                        'abs_positions': positions,
                        'rel_positions': rel_positions if text_id[0] else positions,
                        'relations': []
                    }

                    instances[text_id[0]]['terms'].append(t)

                elif t.startswith('R'): # relation
                    args = args[0].split()
                    label, arg1, arg2 = args
                    arg1 = arg1.split(':')[1]
                    arg2 = arg2.split(':')[1]

                    arg1_text_id = terms[arg1]['text_id']
                    arg2_text_id = terms[arg2]['text_id']

                    assert arg1_text_id == arg2_text_id

                    relations[t] = {
                        'label': label,
                        'tokens': instances[arg1_text_id]['txt'],
                        'subj': terms[arg1],
                        'subj_id': arg1,
                        'obj': terms[arg2],
                        'obj_id': arg2
                    }

                    terms[arg1]['relations'].append(t)
                    terms[arg2]['relations'].append(t)

                    instances[arg1_text_id]['relations'].append(t)

                elif t.startswith('*'): # same-as relation
                    same_as_n += 1
                    t = f"*{same_as_n}"

                    args = args[0].split()
                    label, arg1, arg2 = args

                    arg1_text_id = terms[arg1]['text_id']
                    arg2_text_id = terms[arg2]['text_id']

                    assert arg1_text_id == arg2_text_id

                    relations[t] = {
                        'label': label,
                        'tokens': instances[arg1_text_id]['txt'],
                        'subj': terms[arg1],
                        'subj_id': arg1,
                        'obj': terms[arg2],
                        'obj_id': arg2
                    }

                    terms[arg1]['relations'].append(t)
                    terms[arg2]['relations'].append(t)

                    instances[arg1_text_id]['relations'].append(t)
                    
                else:
                    if verbose:
                        print('Unknown input', line)

        # Generate No-Relation instances
        def get_adj_matrix(instance):
            adj = np.identity(len(instance['terms']))
            term2id = {k:v for v, k in enumerate(instance['terms'])}
            for rel in instance['relations']:
                rel = relations[rel]
                t_subj, t_obj = rel['subj_id'], rel['obj_id']
                adj[term2id[t_subj], term2id[t_obj]] = 1.
            
            return adj

        i = 0
        for instance in instances:
            adj = get_adj_matrix(instance)
            neg_subj, neg_obj = (adj == 0.).nonzero()
            for subj, obj in zip(neg_subj, neg_obj):
                relations[f"NR{i}"] = {
                    'label': 'no-relation',
                    'tokens': instance['txt'],
                    'subj': terms[instance['terms'][subj]],
                    'subj_id': instance['terms'][subj],
                    'obj': terms[instance['terms'][obj]],
                    'obj_id': instance['terms'][obj],
                }
                instance['relations'].append(f"NR{i}")
                i += 1

        assert instances_type in ['sentence', 'entity_pair']

        if instances_type == 'entity_pair':
            entity_pair_instances = [EHealthKDEntityPairInstance(rel, self) for rel in tqdm(relations.values(), total=len(relations))]
            return terms, relations, BatchLoaderEntityPair(entity_pair_instances, batch_size, self.tokenizer.pad_token_id, labels=labels)
        elif instances_type == 'sentence':
            raise NotImplementedError
        else:
            raise ValueError

    def get_train(self, batch_size):
        if not self.train_data:
            train_terms, train_relations, self.train_data = self._read_instances(self.train_path, batch_size, labels=True)
            self.train_terms, self.train_relations = train_terms, train_relations
            if self.add_ensemble_data:
                ensemble_terms, ensemble_relations, ensemble_data = self._read_instances(self.ensemble_path, batch_size, labels=True)
                self.ens_terms, self.ens_relations = ensemble_terms, ensemble_relations
                self.train_data += ensemble_data
            
        return self.train_data

    def get_val(self, batch_size):
        if not self.val_data:
            terms, relations, self.val_data = self._read_instances(self.dev_path, batch_size, labels=True)
            self.val_terms, self.val_relations = terms, relations
        return self.val_data

    def get_test(self, batch_size):
        if not self.test_data:
            terms, relations, self.test_data = self._read_instances(self.test_path, batch_size, labels=True)
            self.test_terms, self.test_relations = terms, relations
        return self.test_data

    def write_output(self, path, partition='dev'):
        if partition == 'train':
            terms, relations = self.train_terms, self.train_relations
        elif partition == 'dev':
            terms, relations = self.val_terms, self.val_relations
        else:
            terms, relations = self.test_terms, self.test_relations
        with open(path, 'wb') as out_f:
            for key, value in terms.items():
                positions = ";".join(f"{str(pos0)} {str(pos1)}" for pos0, pos1 in value['abs_positions'])
                out_str = f"{key}\t{value['ner-label']} {positions}\t{value['text']}\n".encode('utf-8')
                out_f.write(out_str)


    def test(self):
        self._read_instances(self.train_path, 16)


class EHealthKDEntityPairInstance(object):

    def __init__(self, rel_inst, dataset):
        super(EHealthKDEntityPairInstance, self).__init__()

        self.dataset = dataset
        self.rel_inst = rel_inst
        self.relation = self.dataset.rel2id[rel_inst['label']]
        self.tokenizer = self.dataset.tokenizer
        self.result_pos = 0     # Necesary argument for the framework
        subj_start, subj_end = rel_inst['subj']['rel_positions'][0][0], rel_inst['subj']['rel_positions'][-1][-1]
        obj_start, obj_end = rel_inst['obj']['rel_positions'][0][0], rel_inst['obj']['rel_positions'][-1][-1]

        E1S_TOKEN = dataset.tokenizer.encode('[E1S]', add_special_tokens=False)
        E1E_TOKEN = dataset.tokenizer.encode('[E1E]', add_special_tokens=False)
        E2S_TOKEN = dataset.tokenizer.encode('[E2S]', add_special_tokens=False)
        E2E_TOKEN = dataset.tokenizer.encode('[E2E]', add_special_tokens=False)

        tokens = rel_inst['tokens']

        if subj_start < obj_start:
            self.input_tokens = self.tokenizer.encode(
                tokens[:subj_start] + '[E1S]' + tokens[subj_start:subj_end] + '[E1E]' + 
                tokens[subj_end:obj_start] + '[E2S]' + tokens[obj_start:obj_end] + '[E2E]' + tokens[obj_end:],
            add_special_tokens=True)
        else:
            self.input_tokens = self.tokenizer.encode(
                tokens[:obj_start] + '[E2S]' + tokens[obj_start:obj_end] + '[E2E]' + 
                tokens[obj_end:subj_start] + '[E1S]' + tokens[subj_start:subj_end] + '[E1E]' + tokens[subj_end:],
            add_special_tokens=True)

        self.input_tokens = torch.tensor(self.input_tokens)

        # Compute the entity actual positions
        self.ent_pos = [torch.argmax((self.input_tokens == E1S_TOKEN[0]).float()).item(), torch.argmax((self.input_tokens == E2S_TOKEN[0]).float()).item()]
        self.ent_end_pos = [torch.argmax((self.input_tokens == E1E_TOKEN[0]).float()).item(), torch.argmax((self.input_tokens == E2E_TOKEN[0]).float()).item()]

    def get_masked_input(self, probability=.5):
        """ TODO: Finish and test
        """
        masked_input_tokens = self.input_tokens.clone()
        # Mask first entity
        if random.random() < probability:
            masked_input_tokens[self.ent_pos[0]+1:self.ent_end_pos[0]] = self.tokenizer.mask_token_id
        # Mask second entity
        if random.random() < probability:
            masked_input_tokens[self.ent_pos[1]+1:self.ent_end_pos[1]] = self.tokenizer.mask_token_id
        
        return masked_input_tokens

    def evaluate(self, outs):
        predictions = [np.argmax(outs)]
        labels = [self.relation]
        positions = [self.result_pos]

        return predictions, labels, positions

    def generate_output(self, pred=None):
        if pred is None:
            pred = self.relation
        if pred == 0:
            return None

        return f"R\t{self.dataset.id2rel[pred]} Arg1:{self.rel_inst['subj_id']} Arg2:{self.rel_inst['obj_id']}\n"


def main():
    dataset = EHealthKD('data/ehealthkd-2020/', 'xlm-mlm-17-1280')
    dev_data = dataset.get_val(16)

    dataset.write_output('data/ehealthkd-2020/data/submissions/rgem/scenario.ann')
    with open('data/ehealthkd-2020/data/submissions/rgem/scenario.ann', 'ab') as out_f:
        for i, _ in enumerate(dev_data):
            # Do the predictions or whatever
            batch_inst = dev_data.get_instances(i)
            for inst in batch_inst:
                out_str = inst.generate_output()
                if out_str:
                    out_f.write(out_str.encode('utf-8'))



if __name__ == "__main__":
    main()