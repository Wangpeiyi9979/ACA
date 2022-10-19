from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import json
import os

class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def get_parameters(self, mode="numpy", param_dict=None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()

class proto_softmax_layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __distance__(self, rep, rel):
        rep_ = rep.view(rep.shape[0], 1, rep.shape[-1])
        rel_ = rel.view(1, -1, rel.shape[-1])
        dis = (rep_ * rel_).sum(-1)
        return dis

    def __init__(self, sentence_encoder, num_class, id2rel, drop = 0, config = None, rate = 1.0):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(proto_softmax_layer, self).__init__()

        self.config = config
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.output_size
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias = False)
        self.drop = nn.Dropout(drop)
        self.id2rel = id2rel
        self.rel2id = {}
        for id, rel in enumerate(id2rel):
            self.rel2id[rel] = id

    def incremental_learning(self, old_class, add_class):
        weight = self.fc.weight.data
        self.fc = nn.Linear(768, old_class + add_class, bias=False).cuda()
        with torch.no_grad():
            self.fc.weight.data[:old_class] = weight[:old_class]

    def set_memorized_prototypes(self, protos):
        self.prototypes = protos.detach().cuda()
   
    def get_feature(self, sentences):
        rep = self.sentence_encoder(sentences)
        return rep.cpu().data.numpy()

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis
    
    def forward(self, sentences):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(sentences) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep)
        return logits, rep

    def mem_forward(self, rep):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem


class Bert_Encoder(base_model):
#  load_config.attention_probs_dropout_prob = config.monto_drop_ratio
    # load_config.hidden_dropout_prob = config.monto_drop_ratio
    def __init__(self, config, attention_probs_dropout_prob=None, hidden_dropout_prob=None, drop_out=None): 
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # for monto kalo
        if attention_probs_dropout_prob is not None:
            assert hidden_dropout_prob is not None and drop_out is not None
            self.bert_config.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.bert_config.hidden_dropout_prob = hidden_dropout_prob
            config.drop_out = drop_out

        # the dimension for the final outputs
        self.output_size = 768

        self.drop = nn.Dropout(config.drop_out)

        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')

        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + 4)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size*2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)
     
        self.layer_normalization = nn.LayerNorm([self.output_size])


    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        '''
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        '''
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(inputs)[1]
        else:
            # in the entity_marker mode, the representation is generated from the representations of
            #  marks [E11] and [E21] of the head and tail entities.
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])

            # input the sample to BERT
            attention_mask = inputs != 0
            tokens_output = self.encoder(inputs, attention_mask=attention_mask)[0] # [B,N] --> [B,N,H]
            output = []

            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                output.append(instance_output) # [B,N] --> [B,2,H]

            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1) # [B,N] --> [B,H*2]
            
            # the output dimension is [B, H*2], B: batchsize, H: hiddensize
            output = self.drop(output)
            output = self.linear_transform(output)
            output = F.gelu(output)
            output = self.layer_normalization(output)
        return output


