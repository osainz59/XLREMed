import numpy as np

class EarlyStopping(object):
    """ Simple implementation of Early Stopping concept based on the validation loss.

    Usage:
    ```
    >>> early_stopping = EarlyStopping(patience=3, delta=.001)
    
    >>> for epoch in epochs:
    >>>     val_loss = ..
    >>>     if early_stopping(val_loss):
    >>>         break
    ```
    """

    def __init__(self, patience=3, delta=.0, save_checkpoint_fn=None):
        super().__init__()
        
        self.patience = patience
        self.delta = delta
        self.save_checkpoint_fn = save_checkpoint_fn

        self.best_loss = np.inf
        self.best_epoch = -1
        self.steps = 0
        self.epoch = -1

    def __call__(self, val_loss: float, *args, **kwargs) -> bool:
        self.epoch += 1
        diff = (self.best_loss + self.delta) - val_loss
        if diff > 0:
            self.steps = self.patience
            if diff > self.delta:
                self.best_epoch = self.epoch
                self.best_loss = val_loss
                if self.save_checkpoint_fn:
                    self.save_checkpoint_fn(*args, val_loss=self.best_loss, **kwargs)
        else:
            self.steps -= 1

        return self.steps <= 0

    def get_best_epoch(self):
        return self.best_epoch