import torch
import torch.nn
from apex import amp
import os
#import fire
from tqdm import tqdm
from pprint import pprint
import json

from sklearn.metrics import precision_recall_fscore_support

from .model import RGEM, BilinearModel, ConcatModel, XLMForMTBFineTuning, AutoModelForRelationExtraction
from .dataset import Dataset, TACREDEntityPair, TACREDSentence, EHealthKD
from .early_stopping import EarlyStopping

import torch
import torch.nn as nn
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)


from transformers.optimization import AdamW, get_linear_schedule_with_warmup

MODELS = {
    'RGEM': RGEM,
    'Bilinear': BilinearModel,
    'Concat': ConcatModel,
    'MTBFineTune': XLMForMTBFineTuning,
    'default': AutoModelForRelationExtraction
}


class Framework(object):
    """A framework wrapping the Relational Graph Extraction model. This framework allows to train, predict, evaluate,
    saving and loading the model with a single line of code.
    """

    def __init__(self, **config):
        super().__init__()

        self.config = config

        self.grad_acc = self.config['grad_acc'] if 'grad_acc' in self.config else 1
        self.device = torch.device(self.config['device'])
        if isinstance(self.config['model'], str):
            self.model = MODELS[self.config['model']](**self.config) 
        else:
            self.model = self.config['model']

        self.class_weights = torch.tensor(self.config['class_weights']).float() if 'class_weights' in self.config else torch.ones(self.config['n_rel'])
        if 'lambda' in self.config:
            self.class_weights[0] = self.config['lambda']
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device), reduction='mean')
        if self.config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.get_parameters(self.config.get('l2', .01)), lr=self.config['lr'], momentum=self.config.get('momentum', 0), nesterov=self.config.get('nesterov', False))
        elif self.config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.get_parameters(self.config.get('l2', .01)), lr=self.config['lr'])
        elif self.config['optimizer'] == 'AdamW':
            self.optimizer = AdamW(self.model.get_parameters(self.config.get('l2', .01)), lr=self.config['lr'])
        else:
            raise Exception('The optimizer must be SGD, Adam or AdamW')
            

    def _train_step(self, dataset, epoch, scheduler=None):
        print("Training:")
        self.model.train()

        total_loss = 0
        predictions, labels, positions = [], [], []
        precision = recall = fscore = 0.0
        progress = tqdm(enumerate(dataset), desc=f"Epoch: {epoch} - Loss: {0.0} - P/R/F: {precision}/{recall}/{fscore}", total=len(dataset))
        for i, batch in progress:
            # uncompress the batch
            seq, mask, ent, label = batch
            seq = seq.to(self.device)
            mask = mask.to(self.device)
            ent = ent.to(self.device)
            label = label.to(self.device)

            #self.optimizer.zero_grad()
            output = self.model(seq, mask, ent)
            loss = self.loss_fn(output, label)
            total_loss += loss.item()
            
            if self.config['half']:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            if (i+1) % self.grad_acc == 0:
                if self.config.get('grad_clip', False):
                    if self.config['half']:
                        nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.config['grad_clip'])
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.optimizer.step()
                self.model.zero_grad()
                if scheduler:
                    scheduler.step()

            # Evaluate results
            pre, lab, pos = dataset.evaluate(i, output.detach().numpy() if self.config['device'] is 'cpu' else 
                                                output.detach().cpu().numpy())

            predictions.extend(pre)
            labels.extend(lab)
            positions.extend(pos)

            if (i+1) % 10 == 0:
                precision, recall, fscore, _ = precision_recall_fscore_support(np.array(labels), np.array(predictions), 
                                                                               average='micro', labels=list(range(1, self.model.n_rel)))

            progress.set_description(f"Epoch: {epoch} - Loss: {total_loss/(i+1):.3f} - P/R/F: {precision:.2f}/{recall:.2f}/{fscore:.2f}")

        # For last iteration
        #self.optimizer.step()
        #self.optimizer.zero_grad()

        predictions, labels = np.array(predictions), np.array(labels)
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='micro', labels=list(range(1, self.model.n_rel)))
        print(f"Precision: {precision:.3f} - Recall: {recall:.3f} - F-Score: {fscore:.3f}")
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        print(f"[with NO-RELATION] Precision: {precision:.3f} - Recall: {recall:.3f} - F-Score: {fscore:.3f}")

        return total_loss / (i+1)

    def _val_step(self, dataset, epoch):
        print("Validating:")
        self.model.eval()
        
        predictions, labels, positions = [], [], []
        total_loss = 0
        with torch.no_grad():
            progress = tqdm(enumerate(dataset), desc=f"Epoch: {epoch} - Loss: {0.0}", total=len(dataset))
            for i, batch in progress:
                # uncompress the batch
                seq, mask, ent, label = batch
                seq = seq.to(self.device)
                mask = mask.to(self.device)
                ent = ent.to(self.device)
                label = label.to(self.device)

                output = self.model(seq, mask, ent)
                loss = self.loss_fn(output, label)
                total_loss += loss.item()

                # Evaluate results
                pre, lab, pos = dataset.evaluate(i, output.detach().numpy() if self.config['device'] is 'cpu' else 
                                                    output.detach().cpu().numpy())

                predictions.extend(pre)
                labels.extend(lab)
                positions.extend(pos)

                progress.set_description(f"Epoch: {epoch} - Loss: {total_loss/(i+1):.3f}")

        predictions, labels = np.array(predictions), np.array(labels)
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='micro', labels=list(range(1, self.model.n_rel)))
        print(f"Precision: {precision:.3f} - Recall: {recall:.3f} - F-Score: {fscore:.3f}")
        noprecision, norecall, nofscore, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        print(f"[with NO-RELATION] Precision: {noprecision:.3f} - Recall: {norecall:.3f} - F-Score: {nofscore:.3f}")

        return total_loss / (i+1), precision, recall, fscore

    def _save_checkpoint(self, dataset, epoch, loss, val_loss):
        print(f"Saving checkpoint ({dataset.name}.pth) ...")
        PATH = os.path.join('checkpoints', f"{dataset.name}.pth")
        config_PATH = os.path.join('checkpoints', f"{dataset.name}_config.json")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'val_loss': val_loss
        }, PATH)
        with open(config_PATH, 'wt') as f:
            json.dump(self.config, f)

    def _load_checkpoint(self, PATH: str, config_PATH: str):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        with open(config_PATH, 'rt') as f:
            self.config = json.load(f)

        return epoch, loss

    def fit(self, dataset, validation=True, batch_size=1, patience=3, delta=0.):
        """ Fits the model to the given dataset.

        Usage:
        ``` y
        >>> rge = Framework(**config)
        >>> rge.fit(train_data)
        """
        self.model.to(self.device)
        train_data = dataset.get_train(batch_size)

        if self.config['half']:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O2', keep_batchnorm_fp32=True)

        if self.config['linear_scheduler']:
            num_training_steps = int(len(train_data) // self.grad_acc * self.config['epochs'])
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.config.get('warmup_steps', 0), num_training_steps=num_training_steps
            )
        else:
            scheduler = None

        early_stopping = EarlyStopping(patience, delta, self._save_checkpoint)

        for epoch in range(self.config['epochs']):
            self.optimizer.zero_grad()
            loss = self._train_step(train_data, epoch, scheduler=scheduler)
            if validation:
                val_loss, _, _, _ = self._val_step(dataset.get_val(batch_size), epoch)
                if early_stopping(val_loss, dataset=dataset, epoch=epoch, loss=loss):
                    break

        # Recover the best epoch
        path = os.path.join("checkpoints", f"{dataset.name}.pth")
        config_path = os.path.join("checkpoints", f"{dataset.name}_config.json")
        _, _ = self._load_checkpoint(path, config_path)

    def predict(self, dataset, return_proba=False) -> torch.Tensor:
        """ Predicts the relations graph for the given dataset.
        """
        self.model.to(self.device)
        self.model.eval()

        predictions, instances = [], []
        with torch.no_grad():
            progress = tqdm(enumerate(dataset), total=len(dataset))
            for i, batch in progress:
                # uncompress the batch
                seq, mask, ent, label = batch
                seq = seq.to(self.device)
                mask = mask.to(self.device)
                ent = ent.to(self.device)
                label = label.to(self.device)

                output = self.model(seq, mask, ent)
                if not return_proba:
                    pred = np.argmax(output.detach().cpu().numpy(), axis=1).tolist()
                else:
                    pred = output.detach().cpu().numpy().tolist()
                inst = dataset.get_instances(i)

                predictions.extend(pred)
                instances.extend(inst)

        return predictions, instances

    def evaluate(self, dataset: Dataset, batch_size=1) -> torch.Tensor:
        """ Evaluates the model given for the given dataset.
        """
        loss, precision, recall, fscore = self._val_step(dataset.get_val(batch_size), 0)
        return loss, precision, recall, fscore

    def save_model(self, path: str):
        """ Saves the model to a file.

        Usage:
        ``` 
        >>> rge = Framework(**config)
        >>> rge.fit(train_data)

        >>> rge.save_model("path/to/file")
        ```

        TODO
        """
        self.model.save_pretrained(path)
        with open(f"{path}/fine_tunning.config.json", 'wt') as f:
            json.dump(self.config, f, indent=4)

    @classmethod
    def load_model(cls, path: str, config_path: str = None, from_checkpoint=False):
        """ Loads the model from a file.

        Args:
            path: str Path to the file that stores the model.

        Returns:
            Framework instance with the loaded model.

        Usage:
        ```
        >>> rge = Framework.load_model("path/to/model")
        ```

        TODO
        """
        if not from_checkpoint:
            config_path = path + '/fine_tunning.config.json'
            with open(config_path) as f:
                config = json.load(f)
            config['pretrained_model'] = path
            rge = cls(**config)

        else:
            if config_path is None:
                raise Exception('Loading the model from a checkpoint requires config_path argument.')
            with open(config_path) as f:
                config = json.load(f)
            rge = cls(**config)
            rge._load_checkpoint(path, config_path)

        return rge


def train():
    # Load the dataset
    dataset = TACREDEntityPair('data/tacred', 'bert-large-uncased')
    # Define the configuration
    config = {
        'model' : 'Concat',
        'pretrained_model' : 'bert-large-uncased',
        'n_rel' : dataset.get_n_rel(),          # Number of relations
        'vocab_size' : len(dataset.tokenizer),  # Vocab size
        'dropout_p' : .3,                       # Dropout p
        'device': "cuda",                       # Device
        'epochs': 100,                           # Epochs
        'lr': 3e-4,                             # Learning rate
        'half': False,
        'hidden_size': 1024,
        'grad_acc': 16, #16 -> F1: 68 at epoch 10
        #'grad_clip': 5.0
        #'lambda': 0.02
        #'class_weights': dataset.get_class_weights()
    }
    print("Configuration:")
    pprint(config)

    rge = Framework(**config)
    rge.fit(dataset, batch_size=5, patience=3, delta=0.0) # 7

def train_ehealthkd():
    dataset = EHealthKD('data/ehealthkd-2020/', 'xlm-mlm-17-1280', add_ensemble_data=True)
    
    config = {
        'model': 'Concat',
        'pretrained_model': 'xlm-mlm-17-1280',
        'n_rel' : dataset.get_n_rel(),
        'vocab_size' : len(dataset.tokenizer),
        'dropout_p': .2,
        'device': 'cuda',
        'epochs': 100,
        'lr': 0.0003,
        'half': False,
        'hidden_size': 1280,
        'grad_acc': 6
    }
    print("Configuration:")
    pprint(config)

    rge = Framework(**config)
    rge.fit(dataset, batch_size=11, patience=3, delta=.0)

def evaluate_ehealthkd():
    with open('checkpoints/XLMed_gold+silver_config.json') as f:
        config = json.load(f)

    dataset = EHealthKD('data/ehealthkd-2020/', config['pretrained_model'])

    rge = Framework.load_model('checkpoints/XLMed_gold+silver.pth', 'checkpoints/XLMed_gold+silver_config.json', from_checkpoint=True)
    predictions, instances = rge.predict(dataset.get_test(batch_size=8))
    dataset.write_output('data/ehealthkd-2020/data/submissions/rgem/test/run1/scenario3-taskB/scenario.ann', partition='test')
    with open('data/ehealthkd-2020/data/submissions/rgem/test/run1/scenario3-taskB/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))


def evaluate_ehealthkd_taskB():
    with open('checkpoints/EHealthKD_config.json') as f:
        config = json.load(f)

    dataset = EHealthKD('data/ehealthkd-2020/', config['pretrained_model'])

    rge = Framework.load_model('checkpoints/EHealthKD.pth', 'checkpoints/EHealthKD_config.json', from_checkpoint=True)
    predictions, instances = rge.predict(dataset.get_train(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/TASKB/train/run1/scenario3-taskB/scenario.ann', partition='train')
    with open('data/ehealthkd-2020/data/submissions/TASKB/train/run1/scenario3-taskB/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))

    predictions, instances = rge.predict(dataset.get_val(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/TASKB/dev/run1/scenario3-taskB/scenario.ann', partition='dev')
    with open('data/ehealthkd-2020/data/submissions/TASKB/dev/run1/scenario3-taskB/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))

    predictions, instances = rge.predict(dataset.get_test(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/TASKB/test/run1/scenario3-taskB/scenario.ann', partition='test')
    with open('data/ehealthkd-2020/data/submissions/TASKB/test/run1/scenario3-taskB/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))


def submission_ehealthkd():
    
    # Load the model
    with open('checkpoints/XLMR-base-baseline_config.json') as f:
        config = json.load(f)

    rge = Framework.load_model('checkpoints/XLMR-base-baseline.pth', 'checkpoints/XLMR-base-baseline_config.json', from_checkpoint=True)

    # DEV-MAIN
    dataset = EHealthKD('data/ehealthkd-2020/', config['pretrained_model'])
    dataset.test_path = 'data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario1-main/'

    predictions, instances = rge.predict(dataset.get_test(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario1-main/scenario.ann', partition='test')
    with open('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario1-main/scenario.ann', 'ab') as out_f:
       for pred, inst in zip(predictions, instances):
           out_str = inst.generate_output(pred)
           if out_str:
               out_f.write(out_str.encode('utf-8'))

    #DEV-B
    dataset = EHealthKD('data/ehealthkd-2020/', config['pretrained_model'])
    dataset.test_path = 'data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario3-taskB/'

    predictions, instances = rge.predict(dataset.get_test(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario3-taskB/scenario.ann', partition='test')
    with open('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario3-taskB/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))

    # DEV-transfer
    dataset = EHealthKD('data/ehealthkd-2020/', config['pretrained_model'])
    dataset.test_path = 'data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario4-transfer/'

    predictions, instances = rge.predict(dataset.get_test(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario4-transfer/scenario.ann', partition='test')
    with open('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/dev/run1/scenario4-transfer/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))

    # TEST-MAIN
    dataset = EHealthKD('data/ehealthkd-2020/', config['pretrained_model'])
    dataset.test_path = 'data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario1-main/'

    predictions, instances = rge.predict(dataset.get_test(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario1-main/scenario.ann', partition='test')
    with open('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario1-main/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))

    # TEST-B
    dataset = EHealthKD('data/ehealthkd-2020/', config['pretrained_model'])
    dataset.test_path = 'data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario3-taskB/'

    predictions, instances = rge.predict(dataset.get_test(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario3-taskB/scenario.ann', partition='test')
    with open('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario3-taskB/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))

    # TEST-transfer
    dataset = EHealthKD('data/ehealthkd-2020/', config['pretrained_model'])
    dataset.test_path = 'data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario4-transfer/'

    predictions, instances = rge.predict(dataset.get_test(batch_size=32))
    dataset.write_output('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario4-transfer/scenario.ann', partition='test')
    with open('data/ehealthkd-2020/data/submissions/IXA_NER_RE_copy/test/run1/scenario4-transfer/scenario.ann', 'ab') as out_f:
        for pred, inst in zip(predictions, instances):
            out_str = inst.generate_output(pred)
            if out_str:
                out_f.write(out_str.encode('utf-8'))





if __name__ == "__main__":
    #fire.Fire(train)
    #train()
    #evaluate_ehealthkd()
    #submission_ehealthkd()
    evaluate_ehealthkd_taskB()
