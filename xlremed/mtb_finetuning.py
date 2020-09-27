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


optimizers = {
    'SGD': SGD,
    'Adam': Adam
}

def parse():
    parser = argparse.ArgumentParser(description='Matching The Blanks (MTB) finetuning script')

    parser.add_argument('model_folder_path', type=str, 
                        help='Path to the dataset file.')
    parser.add_argument('--pretrained_model', type=str, default='xlm-roberta-base',
                        help='The transformer pretrained model')
    parser.add_argument('--force_preprocess', action='store_true', default=False,
                        help='Force the data preprocessing step.')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size.')
    parser.add_argument('--dev_batch_size', type=int, default=4,
                        help='Validation batch size.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='The optimizer, SGD or Adam.')
    parser.add_argument('--lr', type=float, default=3e-4,
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
    parser.add_argument('--grad_acc', type=int, default=16,
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

    # 'xlm-mlm-17-1280' --batch_size=8 --grad_acc=8 --lr=1e-4 --optimizer=SGD 
    #       --> Epoch: 0 - Lr: 1.0e-04 - Loss: 0.720 - P/R/F: 0.06/0.00/0.00:  22%|███▎           | 2866/12839 [08:09<29:19,  5.67it/s]
    # 'bert-base-multilingual-cased' --batch_size=32 --grad_acc=2 --lr=1e-3 --optimizer=SGD 
    #       --> Epoch: 0 - Lr: 1.0e-03 - Loss: 0.672 - P/R/F: 0.43/0.02/0.04:  62%|█████████▉      | 1993/3210 [06:38<04:19,  4.70it/s]
    
    tokenizer_path = opt.model_folder_path# + 'tokenizer'
    dataset = EHealthKD('data/ehealthkd-2020/', tokenizer_path, add_ensemble_data=opt.ensemble_data)
    opt.model = 'MTBFineTune'
    opt.vocab_size = len(dataset.tokenizer)
    opt.dropout_p = .2
    opt.hidden_size = 768
    opt.pretrained_model = opt.model_folder_path# + 'model'
    opt.n_rel = dataset.get_n_rel()

    config = vars(opt)
    #config['lambda'] = .1

    rge = Framework(**config)
    rge.fit(dataset, batch_size=opt.batch_size, patience=opt.patience, delta=opt.delta)
    name = opt.model_folder_path.split('/')[-1]
    rge.save_model(f'checkpoints/{name}')
    dataset.save_tokenizer(f'checkpoints/{name}')

if __name__ == "__main__":
    opt = parse()
    main(opt)
