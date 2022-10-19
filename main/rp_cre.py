from locale import currency
import logging
import os
import time

import random
import sys
from typing import Optional
from copy import deepcopy
from tqdm import tqdm
import hashlib
import argparse

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import torch.optim as optim
from transformers import BertTokenizer
from data_loader import get_data_loader
from sampler import data_sampler
from utils import select_data, set_seed, Save, get_proto
from config import get_config
from utils import classification_report, f1_score, confusion_matrix_view, get_aca_data

from model.rp_cre import Bert_Encoder, Attention_Memory_Simplified, Softmax_Layer

logger = logging.getLogger(__name__)

def evaluate_strict_model(config, test_data, seen_relations, rel2id, mode="cur", logger=None, pid2name=None, rp_cre_input={}):
    data_loader = get_data_loader(config, test_data, batch_size=256)
    n = len(test_data)
    id2rel = {i: j for j, i in rel2id.items()}
    gold = []
    pred = []
    correct = 0

    encoder = rp_cre_input['encoder']; encoder.eval()
    classifier = rp_cre_input['classifier']; classifier.eval()
    memory_network = rp_cre_input['memory_network']; memory_network.eval()
    protos4eval = rp_cre_input['protos4eval']

    with torch.no_grad():
        for _, (_, labels, tokens, _, _) in enumerate(tqdm(data_loader, desc="Evaluate {}".format(mode))):
            labels = labels.to(config.device)
            tokens = tokens.to(config.device)

            mem_for_batch = protos4eval.clone()
            mem_for_batch.unsqueeze(0)
            mem_for_batch = mem_for_batch.expand(len(tokens), -1, -1)
            reps = encoder(tokens)
            reps = memory_network(reps, mem_for_batch)
            logits = classifier(reps)[:, :config.num_of_relation]
    
            predicts = logits.max(dim=-1)[1]
            labels = labels

            correct += (predicts == labels).sum().item()
            
            predicts = predicts.cpu().tolist()
            labels = labels.cpu().tolist()
            pred_rel_t = [seen_relations.index(id2rel[p]) for p in predicts]
            gold_rel_t = [seen_relations.index(id2rel[l]) for l in labels]
            gold.extend(gold_rel_t)
            pred.extend(pred_rel_t)
        
    micro_f1 = f1_score(gold, pred, average='micro')
    macro_f1 = f1_score(gold, pred, average='macro')
    if logger is not None:
        logger.info('Result {}'.format(mode))
        if len(pid2name) != 0:
            seen_relations = [x+pid2name[x][0] for x in seen_relations]
        logger.info('\n' + classification_report(gold, pred, labels=range(len(seen_relations)), target_names=seen_relations, zero_division=0))
        logger.info("Micro F1 {}".format(micro_f1))
        logger.info("Macro F1 {}".format(macro_f1))
        logger.info(f"confusion matrix\n{confusion_matrix_view(gold, pred, seen_relations, logger)}")

    return correct / n


def train_simple_model(config, encoder, classifier, training_data, epochs):

    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
                            {'params': encoder.parameters(), 'lr': 0.00001},
                            {'params': classifier.parameters(), 'lr': 0.001}
                            ])

    for epoch_i in range(epochs):
        losses = []
        for _, (_, labels, tokens, _, _) in enumerate(data_loader):
            encoder.zero_grad()
            classifier.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)
            logits = classifier(reps)

            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"epoch: {epoch_i}; loss: {np.array(losses).mean()}")


def train_mem_model(config, encoder, classifier, memory_network, training_data, mem_data, epochs):
    data_loader = get_data_loader(config, training_data)
    encoder.train()
    classifier.train()
    memory_network.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001},
        {'params': memory_network.parameters(), 'lr': 0.0001}
    ])

    # mem_data.unsqueeze(0)
    # mem_data = mem_data.expand(data_loader.batch_size, -1, -1)
    for _ in range(epochs):
        losses = []
        for _, (_, labels, tokens, _, _) in enumerate(data_loader):

            mem_for_batch = mem_data.clone()
            mem_for_batch.unsqueeze(0)
            mem_for_batch = mem_for_batch.expand(len(tokens), -1, -1)

            encoder.zero_grad()
            classifier.zero_grad()
            memory_network.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)
            reps = memory_network(reps, mem_for_batch)
            logits = classifier(reps)[:, :config.num_of_relation]

            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(memory_network.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")


def main():


    config = get_config()
    config.exp_name = f'RPCRE'
    if config.aca:
        config.exp_name += '-aca'
        
    config.exp_name += f'-{config.task_name}' + f'-M_{config.memory_size}'

    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    log_path = os.path.join(config.log_dir, "{}".format(config.exp_name) + '.txt')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )
    logger.setLevel(logging.INFO)
    logger.info(config.exp_name)

    tokenizer = BertTokenizer.from_pretrained(config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    test_cur_record = []
    test_total_record = []
    pid2name = json.load(open('data/pid2name.json', 'r')) if  config.task_name.lower() == 'fewrel' else {}
    # set training batch

    for i in range(config.total_round):
       
        test_cur = []
        test_total = []
        set_seed(config.seed + i * 100)

        # sampler setup
        sampler = data_sampler(config=config, seed=config.seed + i * 100, tokenizer=tokenizer)
        
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id



        encoder = Bert_Encoder(config=config).cuda()

        if config.aca:
            add_relation_num = config.rel_per_task * 3
            classifier = Softmax_Layer(input_size=encoder.output_size, num_class=config.num_of_relation + add_relation_num).cuda()
        else:
            classifier = Softmax_Layer(input_size=encoder.output_size, num_class=config.num_of_relation).cuda()

        memorized_samples = {}
        

        # load data and start computation
        for episode, (training_data, _, test_data, current_relations,
                    historic_test_data, seen_relations) in enumerate(sampler):

            print(current_relations)
            temp_mem = {}
            temp_protos = []
            for relation in seen_relations:
                if relation not in current_relations:
                    temp_protos.append(get_proto(config, encoder, memorized_samples[relation]))
                    
            # Initial
            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            if config.aca:
                add_aca_data = get_aca_data(config, deepcopy(training_data), current_relations, tokenizer)
                train_data_for_initial += add_aca_data

            # train model
            random.shuffle(train_data_for_initial)
            logger.info(f'data num for step 1: {len(train_data_for_initial)}')
            train_simple_model(config, encoder, classifier, train_data_for_initial, config.step1_epochs)

            if config.aca:
                logger.info('reset added classifier')
                classifier.incremental_learning(config.num_of_relation, add_relation_num)

            
            for relation in current_relations:
                temp_mem[relation] = select_data(config, encoder, training_data[relation])
                temp_protos.append(get_proto(config, encoder, temp_mem[relation]))

            temp_protos = torch.cat(temp_protos, dim=0).detach()
            memory_network = Attention_Memory_Simplified(mem_slots=len(seen_relations),
                                                input_size=encoder.output_size,
                                                output_size=encoder.output_size,
                                                key_size=config.key_size,
                                                head_size=config.head_size
                                                ).to(config.device)

            # generate training data for the corresponding memory model (ungrouped)
            train_data_for_memory = []
            for relation in temp_mem.keys():
                train_data_for_memory += temp_mem[relation]
            for relation in memorized_samples.keys():
                train_data_for_memory += memorized_samples[relation]
            
            logger.info(f'data num for step 2: {len(train_data_for_memory)}')
            random.shuffle(train_data_for_memory)

            print('[Train Memory Model]')
            train_mem_model(config, encoder, classifier, memory_network, train_data_for_memory, temp_protos, config.step2_epochs)

            # regenerate memory
            print('[Select Memory  Sample]')
            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, training_data[relation])

            print('[Get Prototype of Memory]')
            protos4eval = []
            for relation in memorized_samples:
                protos4eval.append(get_proto(config, encoder, memorized_samples[relation]))
            protos4eval = torch.cat(protos4eval, dim=0).detach()

            print('[Evaluation]')
            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]

            rp_cre_input = {'encoder': encoder, 'classifier': classifier, 'memory_network':memory_network, 'protos4eval': protos4eval}
            cur_acc = evaluate_strict_model(config, test_data_1,seen_relations, rel2id, mode="cur", pid2name=pid2name,  rp_cre_input=rp_cre_input)
            total_acc =  evaluate_strict_model(config, test_data_2 ,seen_relations, rel2id, mode="total", logger=logger, pid2name=pid2name, rp_cre_input=rp_cre_input)


            logger.info(f'Restart Num {i+1}')
            logger.info(f'task--{episode + 1}:')
            test_cur.append(cur_acc)
            test_total.append(total_acc)
            logger.info(f'history test acc:{test_total}')
            logger.info(f'current test acc:{test_cur}')
            


        test_cur_record.append(test_cur)
        test_total_record.append(test_total)


    test_cur_record = torch.tensor(test_cur_record)
    test_total_record = torch.tensor(test_total_record)

    test_cur_record = torch.mean(test_cur_record, dim=0).tolist()
    test_total_record = torch.mean(test_total_record, dim=0).tolist()

    logger.info(f'Avg current test acc: {test_cur_record}')
    logger.info(f'Avg total test acc: {test_total_record}')

    print("log file has been saved in: ", log_path)


if __name__ == "__main__":
    main()