import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, \
    confusion_matrix
from sklearn.preprocessing import label_binarize
import argparse
import json
import os

from framework import Framework
from dataset import EHealthKD, TACREDEntityPair, TACREDSentence

datasets = {
    'eHealthKD': EHealthKD,
    'TACREDEP': TACREDEntityPair,
    'TACREDSnt': TACREDSentence
}

class Evaluator(object):

    def __init__(self, framework: Framework, dataset, 
                batch_size=32):
        super().__init__()

        self.framework = framework
        self.dataset = dataset
        self.batch_size = batch_size
        self.classes = self.dataset.id2rel

    def get_predictions(self, partition='dev', return_proba=False):
        if partition == 'train':
            pred, instances = self.framework.predict(self.dataset.get_train(batch_size=self.batch_size),
                                            return_proba=return_proba)
        elif partition == 'dev':
            pred, instances = self.framework.predict(self.dataset.get_val(batch_size=self.batch_size),
                                            return_proba=return_proba)
        elif partition == 'test':
            pred, instances = self.framework.predict(self.dataset.get_test(batch_size=self.batch_size),
                                            return_proba=return_proba)
        else:
            raise ValueError('partition argument must be train, dev or test')

        labels = np.array([inst.relation for inst in instances])
        pred = np.array(pred)

        return labels, pred

    def get_confusion_matrix(self, y_true, y_pred, plot=False, partition_name=None):
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if plot:
            fig = plt.figure(figsize=(8, 8))
            plt.title(f"{partition_name} partition confusion matrix")
            sns.heatmap(cm, annot=True, fmt=".2f", cbar=False, square=True,
            xticklabels=self.classes, yticklabels=self.classes)
            tick_marks = np.arange(len(self.classes)) + .5
            plt.xticks(tick_marks, self.classes, rotation=35)
            
            return cm, fig

        return cm

    def get_multiclass_precision_recall_curve(self, y_true, y_probs, partition_name=None):
        if len(y_true.shape) == 1:
            y_true = label_binarize(y_true, list(range(len(self.classes))))

        fig = plt.figure(figsize=(10, 10))
        plt.title(f"{partition_name} partition per class precision-recall curve")
        # Plot the F-Scores
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        pre, rec, _ = precision_recall_curve(y_true.ravel(), y_probs.ravel())
        plt.plot(rec, pre, label='micro-averaged', lw=2)

        # Plot the precision recall curve for each class
        for i, label in enumerate(self.classes):
            pre, rec, _ = precision_recall_curve(y_true[:, i],
                                                 y_probs[:, i])
            plt.plot(rec, pre, '-' if i // 8 == 0 else '--', label=label, lw=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        
        return fig



def parse():
    parser = argparse.ArgumentParser(description='Model evaluation script')

    parser.add_argument('checkpoint', type=str, 
                        help='Path to the checkpoint file.')
    parser.add_argument('config', type=str, 
                        help='Path to the config file.')
    parser.add_argument('dataset', type=str, 
                        help='Path to the dataset.')
    parser.add_argument('--dataset_type', type=str, default='eHealthKD',
                        help='Type of the dataset.')
    parser.add_argument('-o', '--output', type=str, dest='output',
                        help='Path to the output folder.')
    parser.add_argument('--partition', type=str, default='dev', 
                        help='Partition of the dataset.')
    parser.add_argument('--from_checkpoint', action='store_true', default=False,
                        help='Load the model from a checkpoint.')


    args = parser.parse_args()
    
    return args

def main(opt):
    # TODO

    os.makedirs(opt.output, exist_ok=True)

    with open(opt.config) as f:
        config = json.load(f)

    dataset = datasets[opt.dataset_type](opt.dataset, config['pretrained_model'])
    rge = Framework.load_model(opt.checkpoint, opt.config, from_checkpoint=opt.from_checkpoint)
    evaluator = Evaluator(rge, dataset)

    model_name = opt.checkpoint.split('.')[0].split('/')[-1]

    y_true, y_probs = evaluator.get_predictions(opt.partition, return_proba=True)
    y_pred = np.argmax(y_probs, axis=1)
    np.save(f"{opt.output}/{opt.partition}-{model_name}-y_true.npy", y_true)
    np.save(f"{opt.output}/{opt.partition}-{model_name}-y_probs.npy", y_probs)

    # Print the general results
    n_rel = len(evaluator.classes)
    labels = list(range(1, n_rel))
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', labels=labels)
    print(f"Precision: {precision:.3f} - Recall: {recall:.3f} - F-Score: {fscore:.3f}")
    noprecision, norecall, nofscore, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print(f"[with NO-RELATION] Precision: {noprecision:.3f} - Recall: {norecall:.3f} - F-Score: {nofscore:.3f}")

    # Get the confusion matrix
    _, cm_f = evaluator.get_confusion_matrix(y_true, y_pred, plot=True, partition_name=opt.partition)
    cm_f.savefig(f"{opt.output}/{opt.partition}-{model_name}-cm.png")

    # Get the precision-recall curve
    pr_f = evaluator.get_multiclass_precision_recall_curve(y_true, y_probs, partition_name=opt.partition)
    pr_f.savefig(f"{opt.output}/{opt.partition}-{model_name}-pr_curve.png")

if __name__ == "__main__":
    opt = parse()
    main(opt)