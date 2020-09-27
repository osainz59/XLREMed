#import fire
from pprint import pprint
from copy import deepcopy

from .framework import Framework
from .dataset import TACRED


class GridSearch(object):
    """ Grid Search algorithm implementation to search for optimal 
        hyperparameter setup.

    TODO: Finish the implementation

    Usage:
    ```
    >>> search = GridSearch(Framework)
    >>> dataset = TACRED('path/to/dataset')
    >>> configurations = {
        ...
    }
    >>> best_solution = search(configurations, dataset)
    ```
    """

    def __init__(self, framework):
        super().__init__()

        self.framework = framework

    def _generate_posible_configurations(self, configurations):
        posible_configurations = [{}]
        for key, values in configurations.items():
            if isinstance(values, list):
                new_dicts = []
                for value in values:
                    for conf in posible_configurations:
                        new_d = deepcopy(conf)
                        new_d[key] = value
                        new_dicts.append(new_d)

                posible_configurations = new_dicts
            else:
                for conf in posible_configurations:
                    conf[key] = values

        return posible_configurations

    def __call__(self, configurations, dataset):
        posible_configurations = self._generate_posible_configurations(configurations)

        best_conf = None
        best_score = (0., 0., 0.)
        scores = []
        for configuration in posible_configurations:
            rgem = self.framework(**configuration)
            rgem.fit(dataset)
            _, prec, rec, f1 = rgem.evaluate(dataset)
            scores.append(
                (configuration, prec, rec, f1)
            )
            if f1 > best_score[-1]:
                best_score = (prec, rec, f1)
                best_conf = configuration



        return best_conf, best_score, scores
        

def test():
    search = GridSearch(Framework)
    dataset = TACRED('data/tacred')

    config = {
        'n_rel' : 42,                   # Number of relations
        'hidden_size' : [200, 300],     # Heads hidden size
        'dropout_p' : .2,               # Dropout p
        'device': "cuda",               # Device
        'epochs': 10,                   # Epochs
        'lr': [2e-5, 2e-4, 2e-3],        # Learning rate
        'l2': [0.01, 0.02, .001],       # L2 normalization
        'lambda': [2e-2, 2e-3, .1]      # No-Relation class weigth
    }

    best_conf, best_score, scores = search(config, dataset)
    print(f"Best configuration with scores: {best_score}")
    pprint(best_conf)

    for conf, pre, rec, f1 in scores:
        print("Configuration:")
        pprint(conf)
        print(f"Precision: {pre}\tRecall: {rec}\tF1-Score: {f1}")


if __name__ == "__main__":
    #fire.Fire(test)
    test()
