import os
import numpy as np
import json
import random
import torch
from transformers import BertTokenizer
from tqdm import tqdm

class data_sampler(object):

    def __init__(self, config=None, seed=None, tokenizer=None):

        self.config = config

        self.tokenizer = tokenizer

        # read relation data
        self.id2rel, self.rel2id = self._read_relations(config.relation_file)

        # random sampling
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
      
        if not os.path.exists(config.cache_file):
            dataset = self._read_data(self.config.data_file)
            torch.save(dataset, config.cache_file)
        else:
            dataset = torch.load(config.cache_file)

        self.training_dataset, self.valid_dataset, self.test_dataset = dataset

        # generate the task number
        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            raise StopIteration()

        indices = self.shuffle_index[self.config.rel_per_task*self.batch: self.config.rel_per_task*(self.batch+1)]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indices:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations


    def _read_data(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for i in range(self.config.num_of_relation)]
        val_dataset = [[] for i in range(self.config.num_of_relation)]
        test_dataset = [[] for i in range(self.config.num_of_relation)]

        # random sample 40 samples for test, and 320 samples for train
        for relation in tqdm(data.keys(), desc="Load data"):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {}
                tokenized_sample['relation'] = self.rel2id[sample['relation']]
                tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']))
                tokenized_sample['string'] = '[RelData] ' + sample['relation'] + ' ' + ' '.join(sample['tokens'])
               
                if self.config.task_name == 'FewRel':
                    if i < self.config.num_of_train:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < self.config.num_of_train + self.config.num_of_val:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:
                            break
        return train_dataset, val_dataset, test_dataset

    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id