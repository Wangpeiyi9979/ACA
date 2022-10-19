from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

@dataclass
class REOutput(SequenceClassifierOutput):
    rel_hidden_states: Optional[torch.FloatTensor] = None
    teacher_probs: Optional[List[torch.FloatTensor]] = None



class BertForRE(BertPreTrainedModel):
    def __init__(self, config, drop=0.5):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(drop)
        self.linear = nn.Linear(768 * 2, 768, bias=True)
        self.layer_normalization = nn.LayerNorm([768])
        self.re_classifier = nn.Linear(768, config.num_labels, bias=False)
        self.init_weights()

    def incremental_learning(self, old_class, add_class):
        weight = self.re_classifier.weight.data
        self.re_classifier = nn.Linear(768, old_class + add_class, bias=False).cuda()
        with torch.no_grad():
            self.re_classifier.weight.data[:old_class] = weight[:old_class]

    def forward(
            self,
            input_ids=None,
            labels=None,
    ):
        attention_mask = input_ids != 0
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        # in the entity_marker mode, the representation is generated from the representations of
        #  marks [E11] and [E21] of the head and tail entities.
        e11 = []
        e21 = []
        # for each sample in the batch, acquire the positions of its [E11] and [E21]
        for i in range(input_ids.size()[0]):
            tokens = input_ids[i].cpu().numpy()
            e11.append(np.argwhere(tokens == 30522)[0][0])
            e21.append(np.argwhere(tokens == 30524)[0][0])

        tokens_output = outputs[0]

        rel_hidden_states = []

        # for each sample in the batch, acquire its representations for [E11] and [E21]
        for i in range(len(e11)):
            instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
            rel_hidden_states.append(instance_output)  # [B,N] --> [B,2,H]

        # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
        rel_hidden_states = torch.cat(rel_hidden_states, dim=0)
        rel_hidden_states = rel_hidden_states.view(rel_hidden_states.size()[0], -1)  # [B,N] --> [B,H*2]

        rel_hidden_states = self.dropout(rel_hidden_states)
        rel_hidden_states = self.linear(rel_hidden_states)
        rel_hidden_states = F.gelu(rel_hidden_states)
        rel_hidden_states = self.layer_normalization(rel_hidden_states)
        
        logits = self.re_classifier(rel_hidden_states)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return REOutput(
            loss=loss,
            logits=logits,
            rel_hidden_states=rel_hidden_states,
        )