import argparse
from .model import XLMForMTBFineTuning
from .dataset import EHealthKD
from .framework import Framework

from torch.optim import SGD, Adam
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)

def parse():
    parser = argparse.ArgumentParser(description='Fine-tuning script')

    parser.add_argument('--pretrained_model', type=str, default='xlm-roberta-base',
                        help='The transformer pretrained model')
    parser.add_argument('--force_preprocess', action='store_true', default=False,
                        help='Force the data preprocessing step.')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size.')
    parser.add_argument('--dev_batch_size', type=int, default=4,
                        help='Validation batch size.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='The optimizer, SGD or Adam.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='Use nesterov momentum')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Momentum')
    parser.add_argument('--l2', type=float, default=0.01,
                        help='L2 normalization')
    parser.add_argument('--mlm_probability', type=float, default=.15,
                        help='Masked Language Model masking probability.')
    parser.add_argument('--linear_scheduler', action='store_true', default=False,
                        help='Whether to use linear scheduler.')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmups of the linear scheduler.')
    parser.add_argument('--mtb_probability', type=float, default=.7,
                        help='Matching The Blanks masking probability.')
    parser.add_argument('--lambd', type=float, default=.7,
                        help='Interpolation weight for the MTB loss.')
    parser.add_argument('--half', action='store_true', default=False,
                        help='Use of half precision training.')
    parser.add_argument('--grad_acc', type=int, default=1,
                        help='Number of steps for gradient accumulation')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of extra iterations for the early stopping.')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Aceptable loss difference between iterations.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug flag. Enable when you are testing changes.')
    parser.add_argument('--ensemble_data', action='store_true', default=False,
                        help='Add extra data on training.')
    parser.add_argument('--recover_training', action='store_true', default=False,
                        help='Continues the previous training.')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Which device to use.')


    args = parser.parse_args()
    
    return args

def main(opt):

    dataset = EHealthKD('data/ehealthkd-2020/', opt.pretrained_model, add_ensemble_data=opt.ensemble_data)
    opt.model = 'MTBFineTune'
    opt.vocab_size = len(dataset.tokenizer)
    opt.dropout_p = .2
    opt.n_rel = dataset.get_n_rel()

    config = vars(opt)

    rge = Framework(**config)
    rge.fit(dataset, batch_size=opt.batch_size, patience=opt.patience, delta=opt.delta)
    name = opt.model_folder_path.split('/')[-1]
    rge.save_model(f'checkpoints/{name}')
    dataset.save_tokenizer(f'checkpoints/{name}')

if __name__ == "__main__":
    opt = parse()
    main(opt)
