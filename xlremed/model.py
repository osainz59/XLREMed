import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings
import os

try:
    from transformers import DistilBertModel, BertModel, AutoModel, AutoConfig, AutoModelWithLMHead
    from transformers.modeling_bert import gelu
except:
    from pytorch_transformers import DistilBertModel, BertModel, AutoModel, AutoConfig
    from pytorch_transformers.modeling_bert import gelu

class RGEM(nn.Module):
    """ Relational Graph Extraction Model.
    """

    def __init__(self, n_rel, vocab_size:int, hidden_size: int, dropout_p: float, **kwargs):
        super().__init__()

        self.n_rel = n_rel
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.encoder.resize_token_embeddings(self.vocab_size)
        self.ent_pooling = SimpleGCN(768, self.hidden_size)
        self.mg_decoder = MultiGraphDecoder(self.n_rel, self.hidden_size, self.dropout_p)

    def forward(self, x, entity_matrix):
        x = self.encoder(x)[0]
        x = self.ent_pooling(x, entity_matrix)
        A = self.mg_decoder(x)

        return A

    def get_parameters(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        return optimizer_grouped_parameters


class SimpleGCN(nn.Module):
    """ Simple GCN implementation without self-loops for Entity Pooling.


    For simplicity:
        - The non-entities representations are going to be masked as 
          zero vectors.
        - Just the first token of the entities with more than one tokens
          will store the entity representation.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.W = nn.Linear(in_features, out_features)
        self.sigma = nn.ReLU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Add an epsilon to avoid divided by zero if the graph is not fully connected
        A_ = A / (A.sum(-1).unsqueeze(-1) + 1e-10)
        xW = self.W(x)
        AxW = A_.bmm(xW)

        return self.sigma(AxW)


class MultiGraphDecoder(nn.Module):
    """ Multi-Graph Decoder.
    """

    def __init__(self, n_graph: int, hidden_size: int, dropout: float = 0.):
        super().__init__()
        assert(n_graph > 0 and hidden_size > 0)

        self.n_graph = n_graph
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        self.head_w = nn.Linear(self.hidden_size, self.hidden_size * self.n_graph)
        self.tail_w = nn.Linear(self.hidden_size, self.hidden_size * self.n_graph)

        self.d = math.sqrt(hidden_size)

    def _reshape(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """ Reshapes the output of (batch_size, seq_length, hidden_size * n_graph) to
        (batch_size, n_graph, seq_length, hidden_size).
        
        From huggingface Transformers Self-Attention implementation.
        """
        new_shape = x.size()[:-1] + (self.n_graph, self.hidden_size)
        x = x.view(*new_shape)

        return x.permute(0, 2, 1, 3)

    def similarity_matrix(self, x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        """ Computes self-attention between all the representations of the sequence.

        Args:
            x:  Tensor (Float) (batch_size, seq_length, hidden_size) Input matrix.

        Returns:
            A:  Tensor (Float) (batch_size, n_graph, seq_length, seq_length) Output similarity matrix.
        """
        #xl = (x**2).sum(-1).sqrt().unsqueeze(-1)
        #yl = (y**2).sum(-1).sqrt().unsqueeze(-1)
        #l = xl.matmul(yl.transpose(-1, -2))
        x = x.matmul(y.transpose(-1, -2))

        return x / self.d

    def forward(self, x: torch.Tensor):
        x = F.dropout(x, self.dropout_p, self.training)

        h = self._reshape(self.head_w(x))
        t = self._reshape(self.tail_w(x))

        A = self.similarity_matrix(h, t)

        return A


class BilinearModel(nn.Module):

    def __init__(self, n_rel, vocab_size, dropout_p, **kwargs):
        super().__init__()

        self.n_rel = n_rel
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p

        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.resize_token_embeddings(self.vocab_size)
        self.clf = BilinearClassifier(768, n_rel)

    def forward(self, x: torch.Tensor, entity_positions: torch.Tensor):
        x = self.encoder(x)[0]
        x = F.dropout(x)

        # entity_positions = (batch_size, 2)
        idx = torch.arange(x.shape[0])
        h = F.dropout(x[idx, entity_positions[:, 0], :])
        t = F.dropout(x[idx, entity_positions[:, 1], :])

        # Classify
        logits = self.clf(h, t)

        return logits

    def get_parameters(self):
        params = [
            {
                'params': self.encoder.parameters(),
                'lr': 2e-5
            },
            {
                'params': self.clf.parameters()
            }
        ]
        return params


class BilinearClassifier(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.head_w = nn.Linear(self.in_features, self.in_features)
        self.tail_w = nn.Linear(self.in_features, self.in_features)
        self.clf = nn.Bilinear(self.in_features, self.in_features, self.out_features)

    def forward(self, h: torch.Tensor, t: torch.Tensor):
        h = F.relu(self.head_w(h))
        t = F.relu(self.tail_w(t))

        logits = self.clf(h, t)

        return logits


class ConcatModel(nn.Module):

    def __init__(self, pretrained_model, n_rel, vocab_size, dropout_p, hidden_size, *args, **kwargs):
        super(ConcatModel, self).__init__()

        self.n_rel = n_rel
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size

        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.encoder.resize_token_embeddings(self.vocab_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.linear = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.clf = nn.Linear(self.hidden_size, self.n_rel)

    def forward(self, x: torch.Tensor, mask: torch.FloatTensor, entity_positions: torch.Tensor):
        x = self.encoder(x, attention_mask=mask)[0]
        #x = F.dropout(x)

        idx = torch.arange(x.shape[0])
        #s = x[idx, 0, :]                          # Sentence representation
        h = x[idx, entity_positions[:, 0], :]     # head representation
        t = x[idx, entity_positions[:, 1], :]     # tail representation

        y = torch.cat([h, t], dim=-1)
        y = self.dropout(y)
        y = F.relu(self.linear(y))
        y = self.clf(y)

        return y

    def get_parameters(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        return optimizer_grouped_parameters


class AutoModelForRelationExtraction(nn.Module):
    """ Relation Extraction model that follows Entity Marker strategy.
    """

    def __init__(self, pretrained_model, *args, n_rel=None, dropout_p=.2, **kwargs):
        super(AutoModelForRelationExtraction, self).__init__()

        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.encoder.resize_token_embeddings(kwargs['vocab_size'])
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.config.n_rel = n_rel
        self.n_rel = n_rel
        self.config.dropout_p = dropout_p
        self.dropout_p = dropout_p
        self.config.vocab_size = kwargs['vocab_size']
        self.re_head = REHead(self.config)
        try:
            self.re_head.load_state_dict(torch.load(pretrained_model + '/re_head.pt'))
        except:
            warnings.warn('Pretrained RE Head not found!')

        self.loss_fn = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(dropout_p)
        self.clf = nn.Linear(self.config.hidden_size, n_rel)

    def forward(self,
        input_ids,
        attention_mask,
        ent_pos,
        labels=None
    ):
        x = self.encoder(input_ids, attention_mask=attention_mask)[0]
        f = self.re_head(x, ent_pos)
        f = self.dropout(f)

        logits = self.clf(f)

        if labels:
            loss = self.loss_fn(logits, labels)
            return loss, logits 
        
        return logits

    def get_parameters(self, l2=0.01):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': l2},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        return optimizer_grouped_parameters

    def save_pretrained(self, folder_path, *args, **kwargs):
        os.makedirs(folder_path, exist_ok=True)
        self.encoder.save_pretrained(folder_path)
        self.config.save_pretrained(folder_path)
        torch.save(self.re_head.state_dict(), folder_path+'/re_head.pt')
        torch.save(self.clf.state_dict(), folder_path+'/re_clf.pt')

    @classmethod
    def from_pretrained(cls, folder_path, *args, **kwargs):
        config = AutoConfig.from_pretrained(folder_path)
        model = cls(folder_path, config.n_rel, config.dropout_p)
        model.clf.load_state_dict(torch.load(folder_path + '/re_clf.pt'))

        return model


class AutoModelForRelationContrastiveClustering(nn.Module):
    """ TODO
    """
    pass
    


class XLMForMTBPreTraining(nn.Module):
    """ XLM For Matching The Blanks PreTraining

    TODO: Implement and test the forward pass
    """

    def __init__(self, pretrained_model_name, config, skip_mlm=False):
        super(XLMForMTBPreTraining, self).__init__()

        self.config = config
        self.skip_mlm = skip_mlm
        # Add the entity markers and BLANKS tokens
        model = AutoModelWithLMHead.from_pretrained(pretrained_model_name)
        self.encoder = model.transformer
        self.encoder.resize_token_embeddings(self.config.vocab_size)

        if not self.skip_mlm:
            self.mlm_head = model.pred_layer
            self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.re_head = REHead(self.config)
        self.mtb_loss_fn = nn.BCEWithLogitsLoss()
        

    def forward(self,
        x_input_ids,
        x_attention_mask,
        x_mlm_labels,
        x_ent_pos,
        y_input_ids,
        y_attention_mask,
        y_mlm_labels,
        y_ent_pos,
        mtb_label
    ):
        # Concatenate the inputs to do single pass
        concat_input_ids = torch.cat([x_input_ids, y_input_ids], dim=0)
        concat_attention_mask = torch.cat([x_attention_mask, y_attention_mask], dim=0)
        concat_mlm_labels = torch.cat([x_mlm_labels, y_mlm_labels], dim=0)
        concat_ent_pos = torch.cat([x_ent_pos, y_ent_pos], dim=0)

        # Encoder pass
        concat_features = self.encoder(concat_input_ids, concat_attention_mask)[0]

        # Matching The Blanks
        re_output = self.re_head(concat_features, concat_ent_pos)
        f_x = re_output[:x_input_ids.size(0)].unsqueeze(dim=1)
        f_y = re_output[x_input_ids.size(0):].unsqueeze(dim=1)
        f = (f_x @ torch.transpose(f_y, -1, -2)).squeeze(-1)

        mtb_loss = self.mtb_loss_fn(f, mtb_label)

        output = (mtb_loss, torch.sigmoid(f))

        # Masked Language Modeling
        if not self.skip_mlm:
            mlm_output = self.mlm_head(concat_features).transpose(-1, -2)
            mlm_loss = self.mlm_loss_fn(mlm_output, concat_mlm_labels)

            output = (mlm_loss, mtb_loss, mlm_output[:x_input_ids.size(0)], mlm_output[x_input_ids.size(0):], torch.sigmoid(f))

        return output

    def get_parameters(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        return optimizer_grouped_parameters

    def save_pretrained(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        self.encoder.save_pretrained(output_folder)
        if hasattr(self, 'mlm_head'):
            torch.save(self.mlm_head.state_dict(), output_folder+'/mlm_head.pt')
        torch.save(self.re_head.state_dict(), output_folder+'/re_head.pt')

    @classmethod
    def from_pretrained(cls, folder_path):
        config = AutoConfig.from_pretrained(folder_path)
        model = cls(folder_path, config)
        model.mlm_head.load_state_dict(torch.load(folder_path + '/mlm_head.pt'))
        model.re_head.load_state_dict(torch.load(folder_path + '/re_head.pt'))

        return model


class XLMForMTBFineTuning(nn.Module):

    def __init__(self, pretrained_model, n_rel, *args, dropout_p=.2, load_mlm=False, **kwargs):
        super(XLMForMTBFineTuning, self).__init__()

        self.n_rel = n_rel
        self.config = AutoConfig.from_pretrained(pretrained_model)# + '/config')
        self.encoder = AutoModel.from_pretrained(pretrained_model)# + '/model')
        self.encoder.resize_token_embeddings(kwargs['vocab_size'])
        self.re_head = REHead(self.config)
        try:
            self.re_head.load_state_dict(torch.load(pretrained_model + '/re_head.pt'))#'/model/re_head.pt'))
        except:
            print('Pretrained RE Head not found!')
        self.loss_fn = nn.CrossEntropyLoss()
        if load_mlm:
            self.mlm_head = MLMHead(self.config)
            self.mlm_head.load_state_dict(torch.load(pretrained_model + '/mlm_head.pt'))#'/model/mlm_head.pt'))
            self.mlm_loss_fn = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(dropout_p)
        self.clf = nn.Linear(self.config.hidden_size, n_rel)
        try:
            self.clf.load_state_dict(torch.load(pretrained_model + '/re_clf.pt'))
        except:
            print('Classifier head not found!')


    def forward(self,
        input_ids,
        attention_mask,
        ent_pos,
        mlm_labels=None,
        labels=None
    ):
        x = self.encoder(input_ids, attention_mask=attention_mask)[0]
        f = self.re_head(x, ent_pos)
        f = self.dropout(f)

        logits = self.clf(f)
        
        output = (logits,)

        if labels:
            loss = self.loss_fn(logits, labels)
            output += (loss,)

        if mlm_labels:
            mlm_logits = self.mlm_head(x).transpose(-1, -2)
            mlm_loss = self.mlm_loss_fn(mlm_logits, mlm_labels)
            output += (mlm_logits, mlm_loss)

        if len(output) == 1:
            output = output[0]

        return output

    def get_parameters(self, l2=0.01):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': l2},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        return optimizer_grouped_parameters

    def save_pretrained(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        self.encoder.save_pretrained(output_folder)
        if hasattr(self, 'mlm_head'):
            torch.save(self.mlm_head.state_dict(), output_folder+'/mlm_head.pt')
        torch.save(self.re_head.state_dict(), output_folder+'/re_head.pt')
        torch.save(self.clf.state_dict(), output_folder+'/re_clf.pt')

    @classmethod
    def from_pretrained(cls, folder_path, config):
        #config = AutoConfig.from_pretrained(folder_path)
        model = cls(folder_path, **config)
        #model.mlm_head.load_state_dict(torch.load(folder_path + '/mlm_head.pt'))
        #model.re_head.load_state_dict(torch.load(folder_path + '/re_head.pt'))
        #model.clf.load_state_dict(torch.load(folder_path + '/re_clf.pt'))

        return model
        


class MLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.dense(x)
        x = gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x)

        return x

class REHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2*config.hidden_size, config.hidden_size)

    def forward(
        self, 
        x,                # First sentence input ids
        ent_pos,        # First sentence entity positions
    ):                  
        idx = torch.arange(x.shape[0])

        # First sentence representation
        h = x[idx, ent_pos[:, 0], :]     # head representation
        t = x[idx, ent_pos[:, 1], :]     # tail representation
        x = torch.cat([h, t], dim=-1)
        x = self.dense(x)

        return x


class NERHead(nn.Module):
    """ TODO
    """

    def __init__(self, ner_labels, dropout_p=.2):
        super().__init__()

        self.ner_labels = ner_labels
        self.dropout_p = dropout_p

        self.linear = nn.Linear(768, self.ner_labels)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, input_mask):
        x *= input_mask


