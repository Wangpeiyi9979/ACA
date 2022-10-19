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

class Softmax_Layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, input_size, num_class):
        """
        Args:
            num_class: number of classes
        """
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=False)

    def incremental_learning(self, old_class, add_class):
        weight = self.fc.weight.data
        self.fc = nn.Linear(768, old_class + add_class, bias=False).cuda()
        with torch.no_grad():
            self.fc.weight.data[:old_class] = weight[:old_class]


    def forward(self, input):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        logits = self.fc(input)
        return logits

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


class Attention_Memory_Simplified(base_model):
    def __init__(self, mem_slots, input_size, output_size, key_size, head_size, num_heads=4):
        super(Attention_Memory_Simplified, self).__init__()
        self.mem_slots = mem_slots

        self.mem_size = input_size
        self.input_size = input_size
        self.output_size = output_size

        self.head_size = head_size
        self.num_heads = num_heads

        # query-key-value
        self.query_size = key_size
        self.key_size = key_size
        self.value_size = self.head_size

        self.q_projector = nn.Linear(self.mem_size, self.num_heads * self.query_size)
        self.q_layernorm = nn.LayerNorm([self.num_heads, self.query_size])

        self.kv_projector = nn.Linear(self.mem_size, self.num_heads*(self.key_size + self.value_size))
        self.k_layernorm = nn.LayerNorm([self.num_heads, self.key_size])
        self.v_layernorm = nn.LayerNorm([self.num_heads, self.value_size])

        # MLP for attention
        self.concatnate_mlp = nn.Linear(self.num_heads*self.value_size, self.mem_size)
        self.concatnate_layernorm = nn.LayerNorm([self.mem_size])
        self.attention_output_layernorm = nn.LayerNorm([self.mem_size])

        self.output_mlp = nn.Linear(self.mem_size, self.output_size)
        self.output_layernorm = nn.LayerNorm([self.output_size])

    def multihead_attention(self, input):

        q = self.q_projector(input)
        q_reshape = q.view(q.shape[0], q.shape[1], self.num_heads, self.query_size)
        q_reshape = self.q_layernorm(q_reshape)
        q_transpose = q_reshape.permute(0, 2, 1, 3)

        kv = self.kv_projector(input)
        kv_reshape = kv.view(kv.shape[0], kv.shape[1], self.num_heads, (self.key_size + self.value_size))
        k_reshape, v_reshape = torch.split(kv_reshape, [self.key_size, self.value_size], dim=-1)
        k_reshape = self.k_layernorm(k_reshape)
        v_reshape = self.v_layernorm(v_reshape)
        k_transpose = k_reshape.permute(0, 2, 1, 3)
        v_transpose = v_reshape.permute(0, 2, 1, 3)

        q_transpose *= (self.key_size ** -0.5)
        # make it [B, H, N, N]
        dot_product = torch.matmul(q_transpose, k_transpose.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)
        # output is [B, H, N, V]
        weighted_output = torch.matmul(weights, v_transpose)
        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]=[batch_size,mem_slots,num_head*head_size]

        output_transpose = weighted_output.permute(0, 2, 1, 3).contiguous()
        output_transpose = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))

        output_transpose = self.concatnate_mlp(output_transpose)
        output_transpose = self.concatnate_layernorm(output_transpose)

        return output_transpose

    def attention_over_memory(self, input, memory):
        # [batch_size, input_size]
        # [batch_size, mem_slot, mem_size]
        input_reshape = input.unsqueeze(dim=1)  #[batch_size,1,mem_size]
        
        memory_plus_input = torch.cat([memory, input_reshape], dim=1) # [batch_size,mem_slot+1,mem_size]

        attention_output = self.multihead_attention(memory_plus_input)
        attention_output = self.attention_output_layernorm(attention_output+memory_plus_input)

        # MLP + ADD + LN
        output = self.output_mlp(attention_output)
        output = F.gelu(output)
        output = self.output_layernorm(output+attention_output)

        return output
        # [batch_size, mem_slot+1,mem_size]

    def forward(self, input, memory):
        output = self.attention_over_memory(input, memory)
        output = output[:, -1, :]
        return output