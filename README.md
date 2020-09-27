# Cross Lingual Relation Extraction in Medical Domain

This repository contains the code used on the Relation Extraction subtask of the [eHealth-KD 2020](https://github.com/knowledge-learning/ehealthkd-2020) shared task by the IXA-NER-RE team. The code of this repository is a mess and is not intended to be used other in stuff rather than testing. We encourage you to use just the `AutoModelForRelationExtraction` class from the [xlremed/model.py](xlremed/model.py) package on your own project.

If you want to use our already pretrained models feel free to ask me via email [osainz006@ehu.eus](osainz006@ehu.eus) or [osainz59@gmail.com](osainz59@gmail.com).


## Use (in python):

```python
import numpy as np

from xlremed.framework import Framework
from xlremed.dataset import EHealthKD
from xlremed.evaluate_model import Evaluator

dataset = EHealthKD('path/to/data', 'path/to/tokenizer')

config = {
    'model': 'default',
    'n_rel': dataset.get_n_rel(),
    'vocab_size': len(dataset.tokenizer),
    'dropout_p': .2,
    ...
}

rge = Framework(**config)
reg.fit(dataset, batch_size=64, patience=3, delta=0.0)

# Save the trained checkpoint
reg.save_model('path/for/checkpoint', 'path/for/train_config')

# Load an already trained checkpoint
reg = Framework.load_model('path/for/checkpoint', 'path/for/train_config')

# Predict new data
predictions, instances = reg.predict(dataset.get_test(batch_size=128))

# Using the evaluator
evaluator = Evaluator(framework=reg, dataset=dataset, batch_size=32)
labels, pred_proba = evaluator.get_predictions('test', return_proba=True)
preditions = np.argmax(y_probs, axis=1)

# Plot the confusion matrix
confusion_matrix, cm_figure = evaluator.get_confusion_matrix(labels, predictions, plot=True, partition_name='test')
# Plot the precision-recall curve
pr_f = evaluator.get_multiclass_precision_recall_curve(labels, pred_proba, partition_name='test')
```

## Use (in CLI):

* Training: [xlremed/finetune.py](xlremed/finetune.py)
```
usage: finetune.py [-h] [--pretrained_model PRETRAINED_MODEL]
                   [--force_preprocess] [--max_seq_length MAX_SEQ_LENGTH]
                   [--batch_size BATCH_SIZE] [--dev_batch_size DEV_BATCH_SIZE]
                   [--epochs EPOCHS] [--optimizer OPTIMIZER] [--lr LR]
                   [--momentum MOMENTUM] [--nesterov] [--grad_clip GRAD_CLIP]
                   [--l2 L2] [--mlm_probability MLM_PROBABILITY]
                   [--linear_scheduler] [--warmup_steps WARMUP_STEPS]
                   [--mtb_probability MTB_PROBABILITY] [--lambd LAMBD]
                   [--half] [--grad_acc GRAD_ACC] [--patience PATIENCE]
                   [--delta DELTA] [--debug] [--ensemble_data]
                   [--recover_training] [--device DEVICE]
```
* Evaluating: [xlremed/evaluate_model.py](xlremed/evaluate_model.py)
```
usage: evaluate_model.py [-h] [-o OUTPUT] [--partition PARTITION]
                         checkpoint config dataset dataset_type
```

## Datasets

Already existing dataset classes for:
* [TACRED](https://nlp.stanford.edu/projects/tacred/) 
* [eHealth-KD 2020](https://github.com/knowledge-learning/ehealthkd-2020)

### Using custom datasets:

In order to use custom dataset you need to create a class for it and implement the following methods:

```python
from xlremed.dataset import Dataset, BatchLoaderEntityPair

class CustomDataset(Dataset):

    def __init__(self, path: str, pretrained_model: str):
        super(CustomDataset, self).__init__(name='CustomDataset', pretrained_model=pretrained_model)
        ...
    
    def get_train(self, batch_size: int):
        ...
        return BatchLoaderEntityPair(instances, batch_size=batch_size, pad_token=pad_token, labels=True)

    def get_val(self, batch_size: int):
        ..
        return BatchLoaderEntityPair(instances, batch_size=batch_size, pad_token=pad_token, labels=True)

    def get_test(self, batch_size: int, labels: bool):
        ..
        return BatchLoaderEntityPair(instances, batch_size=batch_size, pad_token=pad_token, labels=labels)
```

where `instances` is a list of `CustomDatasetInstance` objects:

```python
class CustomDatasetInstance(object):
    def __init__(self, rel_inst, dataset):
        super(CustomDatasetInstance, self).__init__()
        ...
        # Necesary argument for the framework
        self.result_pos = 0     
        # Label information. Relation identifier not the name. For example: 3
        self.relation = 
        # A torch tensor containing the the sequence already tokenized and with the entity markers added
        self.input_tokens = ... 
        # [E1S] and [E2S] token index, for example = [5, 13]
        self.ent_pos = ...
        # [E1E] and [E2E] token index
        self.ent_end_pos = ...

    def evaluate(self, outs):
        # Needed for retrocompatibility. Is going to be removed on the future.
        predictions = [np.argmax(outs)]
        labels = [self.relation]
        positions = [self.result_pos]
```


## Models

This framework is intendeed to be used with the **EntityMarkers** based models. For that there is a model already implemented. You can access it by passing `MTBFineTune` or `AutoModelForRelationExtraction` to the framework. Even if other models can also be used, we strongly recommend you to used those. In case that you want to use any other kind of model you should implemement your custom `BatchLoader` in order to process the required information.

Besides, if you want to follow the **EntityMarkers** strategy, you can use your custom models easily. Your custom model should have the following skeleton:

```python
class CustomModel(torch.nn.Module):

    def __init__(self, pretrained_model, n_rel, *args, dropout_p=.2, **kwargs):
        super(CustomModel, self).__init__()
        ...

    def forward(self,
        input_ids,
        attention_mask,
        ent_pos,
        ...):
        ...
        return logits

    def get_parameters(self, l2=0.01):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': l2},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        return optimizer_grouped_parameters
```

At the time of using just add the following lines:

```python
from xlremed.framework import MODELS

MODELS['CustomModel'] = CustomModel

config = {
    'model': 'CustomModel',
    ...
}
```


## Cite us

If you want to cite this work please use this bibtex:

```bibtex
@inproceedings{ixanerre_ehealthkd2020,
  author    = {Andr{\'{e}}s, Edgar and
              Sainz, Oscar and
              Atutxa, Aitziber and
              Lopez de Lacalle, Oier},
  title     = {{IXA-NER-RE at eHealth-KD Challenge 2020: Cross-Lingual Transfer Learning for Medical Relation Extraction}},
  booktitle = {Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2020)},
  year      = {2020},
}
```
