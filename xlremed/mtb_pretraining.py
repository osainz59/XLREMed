import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
from .model import XLMForMTBPreTraining
from .dataset import Tokenizer
from .early_stopping import EarlyStopping
from typing import Dict, List, Tuple
from tqdm import tqdm
from apex.optimizers import FusedSGD, FusedAdam
from apex import amp
import apex
import argparse

import time

import numpy as np

import gzip
import os

optimizers = {
    'SGD': SGD,
    'Adam': Adam
}

import torch
torch.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)


def parse():
    parser = argparse.ArgumentParser(description='Matching The Blanks (MTB) pretraining script')

    parser.add_argument('file_path', type=str, 
                        help='Path to the dataset file.')
    parser.add_argument('--pretrained_model', type=str, default='xlm-mlm-17-1280',
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
    parser.add_argument('--skip_mlm', action='store_true',
                        help='Discard MLM pretraining.')
    parser.add_argument('--mlm_probability', type=float, default=.15,
                        help='Masked Language Model masking probability.')
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
    parser.add_argument('--recover_training', action='store_true', default=False,
                        help='Continues the previous training.')


    args = parser.parse_args()
    
    return args

def get_additional_special_token_mask(tokens, special_tokens):
    return [1 if token in special_tokens else 0 for token in tokens]

def mask_mtb(tokens, E1S, E1E, E2S, E2E, BLANK, mtb_probability):
    # Compute the markers location
    E1S_pos = torch.argmax((tokens == E1S).float()).item()
    E1E_pos = torch.argmax((tokens == E1E).float()).item()
    E2S_pos = torch.argmax((tokens == E2S).float()).item()
    E2E_pos = torch.argmax((tokens == E2E).float()).item()

    # Create the spans
    mask_range = torch.stack([torch.arange(tokens.shape[1]) for _ in range(tokens.shape[0])])
    E1_spans = (mask_range > E1S_pos) & (mask_range < E1E_pos)
    E2_spans = (mask_range > E2S_pos) & (mask_range < E2E_pos)

    p = torch.bernoulli(torch.full((E1_spans.shape[0], 2), mtb_probability)).bool()
    E1_BLANKS = p[:,0].view(-1, 1) & E1_spans
    E2_BLANKS = p[:,1].view(-1, 1) & E2_spans

    tokens[E1_BLANKS] = BLANK
    tokens[E2_BLANKS] = BLANK

    return tokens

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability: float, mtb_probability: float,
                special_tokens=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    # MTB masking
    if not special_tokens:
        special_tokens = tokenizer.encode("[E1S] [E1E] [E2S] [E2E] [BLANK]", add_special_tokens=False)
    E1S, E1E, E2S, E2E, BLANK = special_tokens
    inputs = mask_mtb(inputs, E1S, E1E, E2S, E2E, BLANK, mtb_probability)

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        (torch.tensor(tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)) | torch.tensor(get_additional_special_token_mask(val, special_tokens))).tolist() for val in labels.tolist() 
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def get_ent_pos(tokens, tokenizer):
    tokens = torch.tensor(tokens)
    E1S, E2S = tokenizer.encode("[E1S] [E2S]", add_special_tokens=False)
    return [torch.argmax((tokens == E1S).float()).item(), torch.argmax((tokens == E2S).float()).item()]

def prepare_data(sentences1, sentences2, labels, tokenizer, max_seq_length):
    sentences1 = tokenizer.batch_encode_plus(sentences1, max_length=max_seq_length) 
    sentences2 = tokenizer.batch_encode_plus(sentences2, max_length=max_seq_length)
    x_input_ids, x_attention_masks, y_input_ids, y_attention_masks, lab = [], [], [], [], []
    for x_input_id, x_attention_mask, y_input_id, y_attention_mask, label \
        in tqdm(zip(sentences1['input_ids'], sentences1['attention_mask'], sentences2['input_ids'], sentences2['attention_mask'],
               labels)):
        if len(x_input_id) >= max_seq_length or len(y_input_id) >= max_seq_length:
            continue
        x_input_ids.append(torch.tensor(x_input_id))
        x_attention_masks.append(torch.tensor(x_attention_mask))
        y_input_ids.append(torch.tensor(y_input_id))
        y_attention_masks.append(torch.tensor(y_attention_mask))
        lab.append(label)

    print(len(x_input_ids))
    x_input_ids = pad_sequence(x_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    x_attention_masks = pad_sequence(x_attention_masks, batch_first=True, padding_value=tokenizer.pad_token_id)
    y_input_ids = pad_sequence(y_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    y_attention_masks = pad_sequence(y_attention_masks, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.tensor(lab).float()

    return x_input_ids, x_attention_masks, y_input_ids, y_attention_masks, labels

def save_preprocessed_data(data, folder_path):
    try:
        os.makedirs(folder_path)
    except:
        pass

    np.save(f"{folder_path}/x_input_ids.npy", data['x_input_ids'].numpy())
    np.save(f"{folder_path}/x_attention_masks.npy", data['x_attention_masks'].numpy())
    np.save(f"{folder_path}/y_input_ids.npy", data['y_input_ids'].numpy())
    np.save(f"{folder_path}/y_attention_masks.npy", data['y_attention_masks'].numpy())
    np.save(f"{folder_path}/labels.npy", data['labels'].numpy())

def load_preprocessed_data(folder_path):
    try:
        output_dict = {
            'x_input_ids': torch.from_numpy(np.load(f"{folder_path}/x_input_ids.npy")),
            'x_attention_masks': torch.from_numpy(np.load(f"{folder_path}/x_attention_masks.npy")),
            'y_input_ids': torch.from_numpy(np.load(f"{folder_path}/y_input_ids.npy")),
            'y_attention_masks': torch.from_numpy(np.load(f"{folder_path}/y_attention_masks.npy")),
            'labels': torch.from_numpy(np.load(f"{folder_path}/labels.npy")).float()
        }
    except:
        return None

    return output_dict


def process_corpus(file_path, tokenizer, force=False, save=True, max_seq_length=512):
    folder_path = ".".join(file_path.split('.')[:-2])

    # Load preprocessed data
    if not force:
        output_dict = {
            'train': load_preprocessed_data(folder_path + '/train'),
            'dev': load_preprocessed_data(folder_path + '/dev')
        }

        if output_dict['train'] and output_dict['dev']:
            print('Data loaded correctly!')
            return output_dict
    else:
        output_dict = {}

    print('Processing raw data.')
    # If not loaded correctly generate the data
    sentences1, sentences2, labels = [], [], []
    with gzip.open(file_path, 'rb') as in_f:
        ignore_first_line = True
        for i, line in tqdm(enumerate(in_f)):
            if not i:
                continue
            s1, s2, label = line.decode('utf-8').rstrip().split('\t')[-3:]
            sentences1.append(s1)
            sentences2.append(s2)
            labels.append(int(label))

    # Separate into train and dev
    labels_t = torch.tensor(labels).float()
    pos_length = (labels_t == 1.).sum().item()

    # Train data 80%
    train_sentences1 = sentences1[:int(pos_length*.8)] + sentences1[int(pos_length):int(pos_length*1.8)]
    train_sentences2 = sentences2[:int(pos_length*.8)] + sentences2[int(pos_length):int(pos_length*1.8)]
    train_labels = labels[:int(pos_length*.8)] + labels[int(pos_length):int(pos_length*1.8)]

    x_input_ids, x_attention_masks, y_input_ids, y_attention_masks, labels_ = prepare_data(train_sentences1, train_sentences2, train_labels, tokenizer,
                                                                                           max_seq_length)

    output_dict['train'] = {
        'x_input_ids': x_input_ids,
        'x_attention_masks': x_attention_masks,
        'y_input_ids': y_input_ids,
        'y_attention_masks': y_attention_masks,
        'labels': labels_
    }

    if save:
        save_preprocessed_data(output_dict['train'], folder_path + '/train')

    # Dev data 20%
    dev_sentences1 = sentences1[int(pos_length*.8):int(pos_length)] + sentences1[int(pos_length*1.8):]
    dev_sentences2 = sentences2[int(pos_length*.8):int(pos_length)] + sentences2[int(pos_length*1.8):]
    dev_labels = labels[int(pos_length*.8):int(pos_length)] + labels[int(pos_length*1.8):]

    print(len(dev_sentences1))

    x_input_ids, x_attention_masks, y_input_ids, y_attention_masks, labels_ = prepare_data(dev_sentences1, dev_sentences2, dev_labels, tokenizer,
                                                                                           max_seq_length)

    output_dict['dev'] = {
        'x_input_ids': x_input_ids,
        'x_attention_masks': x_attention_masks,
        'y_input_ids': y_input_ids,
        'y_attention_masks': y_attention_masks,
        'labels': labels_
    }

    if save:
        save_preprocessed_data(output_dict['dev'], folder_path + '/dev')

    return output_dict


def pretrain_model(file_path, opt):

    # Create the folder containing the resulting stuff
    folder_path = ".".join(file_path.split('.')[:-2])
    file_name = folder_path.split('/')[-1]
    try:
        os.makedirs(folder_path+'/model')
        os.makedirs(folder_path+'/config')
        os.makedirs(folder_path+'/tokenizer')
    except:
        pass

    # Define the model and variables
    config = AutoConfig.from_pretrained(opt.pretrained_model)
    tokenizer = Tokenizer.from_pretrained(opt.pretrained_model)

    # Add the additional tokens
    additional_tokens = ['[E1S]', '[E1E]', '[E2S]', '[E2E]', '[BLANK]']
    added_tokens = tokenizer.add_tokens(additional_tokens)
    if len(tokenizer) % 8 == 0:
        vocab_size = len(tokenizer)
    else:
        vocab_size = int((len(tokenizer) // 8 + 1)*8)
    config.vocab_size = vocab_size      # Avoid the extra added tokens by the Roberta tokenizer
    config.save_pretrained(folder_path+'/config')
    tokenizer.save_pretrained(folder_path+'/tokenizer')
    special_tokens = tokenizer.encode("[E1S] [E1E] [E2S] [E2E] [BLANK]", add_special_tokens=False)

    # Define the model
    model = XLMForMTBPreTraining(opt.pretrained_model, config, skip_mlm=opt.skip_mlm)

    if opt.recover_training:
        try:
            model_ = XLMForMTBPreTraining.from_pretrained(folder_path)
        except:
            print("Error at loading previous checkpoint.")
            model_ = model

        model = model_

    # Load the data (process it in case it is not already processed)
    data = process_corpus(file_path, tokenizer, force=opt.force_preprocess, max_seq_length=opt.max_seq_length)

    train_dataset = TensorDataset(
        data['train']['x_input_ids'], data['train']['x_attention_masks'],
        data['train']['y_input_ids'], data['train']['y_attention_masks'],
        data['train']['labels']
    )
    dev_dataset = TensorDataset(
        data['dev']['x_input_ids'], data['dev']['x_attention_masks'],
        data['dev']['y_input_ids'], data['dev']['y_attention_masks'],
        data['dev']['labels']
    )
    del data

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.dev_batch_size, shuffle=True)

    def save_checkpoint_fn(*args, **kwargs):
        model.encoder.save_pretrained(folder_path+'/model')
        if hasattr(model, 'mlm_head'):
            torch.save(model.mlm_head.state_dict(), folder_path+'/model/mlm_head.pt')
        torch.save(model.re_head.state_dict(), folder_path+'/model/re_head.pt')

    # Prepare for training
    early_stopping = EarlyStopping(patience=opt.patience, delta=opt.delta,
                                   save_checkpoint_fn=save_checkpoint_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optimizers[opt.optimizer](model.get_parameters(), opt.lr)
    if opt.half:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)
    if torch.cuda.device_count() > 1:
        if opt.half:
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
    

    def prepare_batch(batch):
        # Uncompress the batch
        x_input_ids, x_attention_masks, y_input_ids, y_attention_masks, \
            labels = batch
        x_input_ids, x_mlm_labels = mask_tokens(x_input_ids, tokenizer, opt.mlm_probability,
                                         opt.mtb_probability, special_tokens=special_tokens)
        y_input_ids, y_mlm_labels = mask_tokens(y_input_ids, tokenizer, opt.mlm_probability,
                                         opt.mtb_probability, special_tokens=special_tokens)
        x_ent_pos = torch.tensor([torch.argmax((x_input_ids == special_tokens[0]).float(), -1).tolist(), 
                                    torch.argmax((x_input_ids == special_tokens[2]).float(), -1).tolist()]).T
        y_ent_pos = torch.tensor([torch.argmax((y_input_ids == special_tokens[0]).float(), -1).tolist(), 
                                    torch.argmax((y_input_ids == special_tokens[2]).float(), -1).tolist()]).T

        # Put into the device
        x_input_ids = x_input_ids.to(device)
        x_attention_masks = x_attention_masks.to(device)
        x_mlm_labels = x_mlm_labels.to(device)
        x_ent_pos = x_ent_pos.to(device)
        y_input_ids = y_input_ids.to(device)
        y_attention_masks = y_attention_masks.to(device)
        y_mlm_labels = y_mlm_labels.to(device)
        y_ent_pos = y_ent_pos.to(device)
        labels = labels.unsqueeze(1).to(device)

        return x_input_ids, x_attention_masks, x_mlm_labels, x_ent_pos, \
               y_input_ids, y_attention_masks, y_mlm_labels, y_ent_pos, \
               labels


    # Training loop
    try:
        for epoch in range(opt.epochs):
            total_loss = mtb_total_loss = mlm_total_loss = 0.
            optimizer.zero_grad()
            model.train()

            mlm_pred, mlm_labels, mtb_pred, mtb_labels = [], [], [], []
            progress = tqdm(enumerate(train_dataloader), desc=f"Epoch: {epoch} - Loss: {0.0}", total=len(train_dataloader))
            for i, batch in progress:
                batch = prepare_batch(batch)
                output = model(*batch)
                if not opt.skip_mlm:
                    mlm_loss, mtb_loss, x_pred, y_pred, f = output
                    loss = opt.lambd*mtb_loss + (1. - opt.lambd)*mlm_loss
                    mtb_total_loss += mtb_loss.item()
                    mlm_total_loss += mlm_loss.item()
                else:
                    loss, f = output
                    mtb_total_loss += loss.item()

                total_loss += loss.item()

                if opt.half:
                    with amp.scale_loss(loss, optimizer) as scale_loss:
                        scale_loss.backward()
                else:
                    loss.backward()

                if (i+1) % opt.grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                progress.set_description(f"Epoch: {epoch} - Loss: {total_loss/(i+1):.3f} - MTBLoss: {mtb_total_loss/(i+1):.3f} - MLMLoss: {mlm_total_loss/(i+1):.3f}")

            # Evaluate
            model.eval()
            with torch.no_grad():
                mlm_pred, mlm_labels, mtb_pred, mtb_labels = [], [], [], []
                total_loss = mtb_total_loss = mlm_total_loss = 0.
                progress = tqdm(enumerate(dev_dataloader), desc=f"Epoch: {epoch} - Loss: {0.0}", total=len(dev_dataloader))
                for i, batch in progress:

                    batch = prepare_batch(batch)
                    output = model(*batch)
                    if not opt.skip_mlm:
                        mlm_loss, mtb_loss, x_pred, y_pred, f = output
                        loss = opt.lambd*mtb_loss + (1. - opt.lambd)*mlm_loss
                        mtb_total_loss += mtb_loss.item()
                        mlm_total_loss += mlm_loss.item()
                    else:
                        loss, f = output
                        mtb_total_loss += loss.item()

                    total_loss += loss.item()

                    progress.set_description(f"Epoch: {epoch} - Loss: {total_loss/(i+1):.3f} - MTBLoss: {mtb_total_loss/(i+1):.3f} - MLMLoss: {mlm_total_loss/(i+1):.3f}")
                

            if early_stopping(total_loss):
                break
    
    except KeyboardInterrupt:
        if not opt.debug:
            print(f"Interrupted! Saving the model in: {folder_path+'/interrupted_model'}")
            try:
                os.makedirs(folder_path+'/interrupted_model')
            except:
                pass

            model.encoder.save_pretrained(folder_path+'/interrupted_model')
            if hasattr(model, 'mlm_head'):
                torch.save(model.mlm_head.state_dict(), folder_path+'/interrupted_model/mlm_head.pt')
            torch.save(model.re_head.state_dict(), folder_path+'/interrupted_model/re_head.pt')




def test2():
    config = AutoConfig.from_pretrained('xlm-mlm-17-1280')
    tokenizer = AutoTokenizer.from_pretrained('xlm-mlm-17-1280')
    tokenizer.add_tokens(['[BLANK]', '[E1S]', '[E1E]', '[E2S]', '[E2E]'])
    #model = XLMForMTBPreTraining(config)

    data = process_corpus('data/medline/esmedline_mtb.lemma.tab.gz', tokenizer)


def test1():
    # Load stuff
    config = AutoConfig.from_pretrained('xlm-mlm-17-1280')
    config.vocab_size += 5
    tokenizer = AutoTokenizer.from_pretrained('xlm-mlm-17-1280')
    tokenizer.add_tokens(['[BLANK]', '[E1S]', '[E1E]', '[E2S]', '[E2E]'])
    model = XLMForMTBPreTraining(config)

    # Prepare an example
    sent1 = "Spontaneous [E2S]epistaxis[E2E] is one of the most frequent [E1S]problems[E1E] in emergency services."
    sent2 = "La [E2S]epistaxis[E2E] espontánea es uno de los [E1S]problemas[E1E] más frecuentes en consultas de urgencia."

    sent1 = "We concluded it is not clear whether topical [E2S]tranexamic acid[E2E] has any impact on hemostasis or risk of rebleeding because the [E1S]certainty[E1E] of the evidence is very low"
    sent2 = "Sin embargo, su rol en el manejo de la [E2S]epistaxis[E2E] espontánea sigue siendo poco claro, existiendo controversia en cuanto a su efectividad y [E1S]seguridad[E1E]"

    x = tokenizer.batch_encode_plus([sent1])
    y = tokenizer.batch_encode_plus([sent1])

    x_input_ids = torch.tensor(x['input_ids'])
    y_input_ids = torch.tensor(y['input_ids'])

    x_attention_mask = torch.tensor(x['attention_mask'])
    y_attention_mask = torch.tensor(y['attention_mask'])

    x_ent_pos = torch.tensor([get_ent_pos(x['input_ids'], tokenizer)])
    y_ent_pos = torch.tensor([get_ent_pos(y['input_ids'], tokenizer)])

    x_mlm_labels = torch.ones_like(x_input_ids) * -100
    y_mlm_labels = torch.ones_like(y_input_ids) * -100

    mtb_labels = torch.tensor([[1.]])

    output = model(
        x_input_ids, 
        x_attention_mask, 
        x_mlm_labels, 
        x_ent_pos, 
        y_input_ids, 
        y_attention_mask, 
        y_mlm_labels, 
        y_ent_pos, 
        mtb_labels
    )

    print(output)

    print("##########")
    print(mask_tokens(x_input_ids, tokenizer, .05, .7))




if __name__ == "__main__":
    opt = parse()
    pretrain_model(opt.file_path, opt)
    #test1()