import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
import time
import json
import os
from copy import deepcopy
import argparse
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from model.emar import Bert_Encoder, proto_softmax_layer
from data_loader import get_data_loader
from sampler import data_sampler
from transformers import BertTokenizer
from sklearn.metrics import f1_score, classification_report
from config import get_config
from utils import save_representation_to_file

import logging
from utils import set_seed, Save, confusion_matrix_view, select_data, get_proto, get_aca_data
logger = logging.getLogger(__name__)


def evaluate_strict_model(config, test_data, seen_relations, rel2id, mode="cur", logger=None, pid2name=None, model=None):

    model.eval()
    n = len(test_data)
    data_loader = get_data_loader(config, test_data, batch_size=128)
    gold = []
    pred = []
    correct = 0

    with torch.no_grad():
        seen_relation_ids = [rel2id[rel] for rel in seen_relations]
        for _, (_, labels, sentences, _, _) in enumerate(data_loader):
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            _, rep = model(sentences)
            logits = model.get_mem_feature(rep)
            predicts = logits.max(dim=-1)[1].cpu()
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()
            correct += (predicts == labels).sum().item()
            predicts = predicts.tolist()
            labels = labels.tolist()

            gold.extend(labels)
            pred.extend(predicts)
    
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

def train_simple_model(config, model, train_set, epochs, step2=False):
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
                            {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
                            {'params': model.fc.parameters(), 'lr': 0.001}
                            ])
    
    for epoch_i in range(epochs):
        losses = []
        for step, (_, labels, sentences, _, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            logits, _ = model(sentences)
            if step2 is True:
                logits = logits[:, :config.num_of_relation]
            labels = labels.cuda()
            loss = criterion(logits, labels)
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        print (np.array(losses).mean())
    return model

def train_model(config, model, mem_set, epochs, current_proto, seen_relation_ids):
    data_loader = get_data_loader(config, mem_set, shuffle=True)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
                            {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
                            {'params': model.fc.parameters(), 'lr': 0.001}
                            ])
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses = []
        for step, (_, labels, sentences, _, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            _, rep = model(sentences)
            logits_proto = model.mem_forward(rep)
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()
            labels = labels.cuda()
            loss = (criterion(logits_proto, labels))
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    return model

if __name__ == '__main__':
    config = get_config()

    config.exp_name = f'EMAR'
    if config.aca:
        config.exp_name += '-aca'
    config.exp_name += f'-{config.task_name}-M_{config.memory_size}'

    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    log_path = os.path.join(config.log_dir, "{}".format(config.exp_name) + '.txt')

    if not os.path.exists(f'reps/{config.exp_name}'):
        os.mkdir(f'reps/{config.exp_name}')

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
    for i in range(config.total_round):
        if not os.path.exists(f'reps/{config.exp_name}/{i}'):
            os.mkdir(f'reps/{config.exp_name}/{i}')

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
            model = proto_softmax_layer(
            encoder, 
            num_class = len(sampler.id2rel) + add_relation_num, 
            id2rel = sampler.id2rel, 
            drop = 0, 
            config = config).cuda()
        else:
            model = proto_softmax_layer(
                encoder, 
                num_class = len(sampler.id2rel), 
                id2rel = sampler.id2rel, 
                drop = 0, 
                config = config).cuda()

        memorized_samples = {}
        

        for episode, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            print(current_relations)
          
            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]
            # Step1-2: Initialize: Just Random Initialize the as other model
            
            if config.aca:
                add_aca_data = get_aca_data(config, deepcopy(training_data), current_relations, tokenizer)

            # Step3 -> Step5
            if config.aca:
                logger.info(f'data num for step 1: {len(train_data_for_initial + add_aca_data)}')
                model = train_simple_model(config, model, train_data_for_initial + add_aca_data, 2) 
            else:
                logger.info(f'data num for step 1: {len(train_data_for_initial)}')
                model = train_simple_model(config, model, train_data_for_initial, 2) 
                
            if config.aca:
                logger.info('remove aca node')
                model.incremental_learning(config.num_of_relation, add_relation_num)

            # Step6
            logger.info('Selecting Examples for Memory')
            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, training_data[relation])

            # Step7
            mem_data = []
            for rel in memorized_samples:
                mem_data += memorized_samples[rel]
            
            
            # Step8: Ak
            data4step2 = mem_data + train_data_for_initial 

            logger.info('Replay, Activation and Reconsolidation')
            seen_relation_ids = [rel2id[rel] for rel in seen_relations]

            # Step9
            for _ in range(2):

                # Step10 - 12 (use all to compute proto)
                protos4train = []
                for relation in seen_relations:
                    protos4train.append(get_proto(config, encoder, memorized_samples[relation]))
                protos4train = torch.cat(protos4train, dim=0).detach()

                # Step13 - 15
                print('Memory Replay and Activation')
                model = train_simple_model(config, model, data4step2, 1) # Memory Replay and Activation

                # Step16 - 18 use all Memory Example; balance
                print("Memory Reconsolidation")
                model = train_model(config, model, mem_data, 1, protos4train, seen_relation_ids) # Memory Reconsolidaton

            protos4eval = []
            for relation in seen_relations:
                r = model.fc.weight[rel2id[relation]].detach()
                proto = get_proto(config, encoder, memorized_samples[relation], r)
                proto = proto / proto.norm()
                protos4eval.append(proto)
            protos4eval = torch.cat(protos4eval, dim=0).detach()

            model.set_memorized_prototypes(protos4eval)

            print('[Evaluation]')
            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]
            cur_acc = evaluate_strict_model(config, test_data_1,seen_relations, rel2id, mode="cur", pid2name=pid2name, model=model)
            total_acc =  evaluate_strict_model(config, test_data_2 ,seen_relations, rel2id, mode="total", logger=logger, pid2name=pid2name, model=model)
            
            save_representation_to_file(config, model, sampler, id2rel, f'reps/{config.exp_name}/{i}/{episode}.pt' ,memorized_samples)

          
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
