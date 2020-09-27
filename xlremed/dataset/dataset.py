import torch
#import fire

from random import shuffle
from .tokenizer import Tokenizer

try:
    from transformers import DistilBertTokenizer, BertTokenizer, AutoTokenizer
except:
    from pytorch_transformers import DistilBertTokenizer, BertTokenizer, AutoTokenizer


class Dataset(object):
    """ An abstract class to deal with different datasets.
    """

    def __init__(self, name, pretrained_model=None):
        super().__init__()

        #assert type in ['sentence', 'entity-pair']

        self.name = name
        #self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.tokenizer = Tokenizer.from_pretrained(pretrained_model)
        #self.tokenizer.add_special_tokens({'additional_special_tokens': ['[E1S]','[E1E]', '[E2S]', '[E2E]']})
        self.tokenizer.add_tokens(['[E1S]','[E1E]', '[E2S]', '[E2E]'])

    def token2wordpiece(self, tokens):
        new_tokens = [self.tokenizer.cls_token_id]
        token_positions = [1]
        for token in tokens:
            new_token = self.tokenizer.encode(token, add_special_tokens=False)
            token_positions.append(token_positions[-1] + len(new_token))
            new_tokens.extend(new_token)
        new_tokens.append(self.tokenizer.sep_token_id)

        return torch.tensor(new_tokens), token_positions

    def get_train(self):
        raise NotImplementedError

    def get_val(self):
        raise NotImplementedError

    def get_test(self):
        raise NotImplementedError

    def adjacency2relations(self, adj):
        return adj.nonzeros()

    def get_n_rel(self):
        return len(self.rel2id)

    def get_class_weights(self):
        return self.class_weights

    def convert2labels(self, pred):
        return [self.id2rel[p] for p in pred]

    def save_tokenizer(self, path: str):
        self.tokenizer.save_pretrained(path)


class BatchLoaderEntityPair(object):
    """ TODO: Test entity masking
    """

    def __init__(self, instances, batch_size, pad_token, labels=True, mask_p=0.):
        super(BatchLoaderEntityPair, self).__init__()

        self.instances = instances
        self.batch_size = batch_size
        self.pad_token = pad_token
        self.n_batch = (len(instances) // batch_size) + (len(instances) % batch_size > 0)
        self.return_labels = labels
        self.mask_p = mask_p

    def get_batch(self, i):
        assert i < self.n_batch
        batch_range = (i*self.batch_size, (i+1)*self.batch_size)
        if self.mask_p > 0.:
            sequences = torch.nn.utils.rnn.pad_sequence([inst.get_masked_input(self.mask_p) for inst in self.instances[batch_range[0]:batch_range[1]]],
                                                     batch_first=True, padding_value=self.pad_token)

        else:
            sequences = torch.nn.utils.rnn.pad_sequence([inst.input_tokens for inst in self.instances[batch_range[0]:batch_range[1]]],
                                                     batch_first=True, padding_value=self.pad_token)
        attention_mask = (sequences != self.pad_token).float()
        ent_pos = torch.tensor([inst.ent_pos for inst in self.instances[batch_range[0]:batch_range[1]]])

        if self.return_labels:
            labels = torch.tensor([inst.relation for inst in self.instances[batch_range[0]:batch_range[1]]])
            return sequences, attention_mask, ent_pos, labels
        else:
            return sequences, attention_mask, ent_pos


    def evaluate(self, i, output):
        predicted, label, position = [], [], []
        batch_range = (i*self.batch_size, (i+1)*self.batch_size)
        for inst, out in zip(self.instances[batch_range[0]:batch_range[1]], output):
            pre, lab, pos = inst.evaluate(out)
            predicted.extend(pre)
            label.extend(lab)
            position.extend(pos)
        return predicted, label, position

    def get_instances(self, i):
        batch_range = (i*self.batch_size, (i+1)*self.batch_size)
        return self.instances[batch_range[0]:batch_range[1]]

    def __iter__(self):
        """ Iterator over batches
        """
        shuffle(self.instances)
        for i in range(self.n_batch):
            yield self.get_batch(i)

    def __len__(self):
        return self.n_batch

    def __add__(self, other: 'BatchLoaderEntityPair'):
        if not isinstance(other, BatchLoaderEntityPair):
            raise TypeError('The addition must be between two BatchLoaderEntityPair objects.')

        self.instances += other.instances
        self.n_batch = (len(self.instances) // self.batch_size) + (len(self.instances) % self.batch_size > 0)

        return self


class BatchLoaderSentence(object):

    def __init__(self, instances, batch_size, pad_token, labels=True):
        super(BatchLoaderSentence, self).__init__()

        self.instances = instances
        self.batch_size = batch_size
        self.pad_token = pad_token
        self.n_batch = (len(instances) // batch_size) + (len(instances) % batch_size > 0)
        self.return_labels = labels

    def get_batch(self, i):
        assert i < self.n_batch
        batch_range = (i*self.batch_size, (i+1)*self.batch_size)
        sequences = torch.nn.utils.rnn.pad_sequence([inst.input_tokens for inst in self.instances[batch_range[0]:batch_range[1]]],
                                                     batch_first=True, padding_value=self.pad_token)
        seq_len = sequences.size(-1)
        entity_matrices = torch.stack([inst.get_entity_matrix(seq_len) for inst in self.instances[batch_range[0]:batch_range[1]]])

        if self.return_labels:
            labels = torch.stack([inst.get_label_matrix(seq_len) for inst in self.instances[batch_range[0]:batch_range[1]]])

            return sequences, entity_matrices, labels
        else:
            return sequences, entity_matrices

    def evaluate(self, i, output):
        predicted, label, position = [], [], []
        batch_range = (i*self.batch_size, (i+1)*self.batch_size)
        for inst, out in zip(self.instances[batch_range[0]:batch_range[1]], output):
            pre, lab, pos = inst.evaluate(out)
            predicted.extend(pre)
            label.extend(lab)
            position.extend(pos)
        return predicted, label, position

    def __iter__(self):
        """ Iterator over batches
        """
        shuffle(self.instances)
        for i in range(self.n_batch):
            yield self.get_batch(i)

    def __len__(self):
        return self.n_batch


def test():
    sentence = "I eat apples in Lasarte .".split()
    positions = [(4,5)]

    dataset = Dataset('test')
    new_sent, new_pos = dataset.token2wordpiece(sentence)

    print(new_pos)

if __name__ == "__main__":
    #fire.Fire(test)
    test()
