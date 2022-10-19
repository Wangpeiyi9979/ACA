import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import torch
import pickle
import re
import random

import sys
plt.rcParams['font.family'] = ['Times New Roman']

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(2021)

N_iter = 2000
head_line = "P156foll | P84archi | P39posit | P276loca | P410mili | P241mili | P177cros | P264reco | P412voic | P361part | P1923par | P123publ | P118leag | P131loca | P710part | P355subs | P6head o | P400plat | P101fiel | P140reli | P2094com | P364orig | P150cont | P466occu | P449orig | P674char | P991succ | P495coun | P1346win | P750dist | P106occu | P25mothe | P706loca | P127owne | P413posi | P463memb | P1408lic | P59const | P86compo | P740loca | P206loca | P17count | P136genr | P800nota | P4552mou | P1001app | P931plac | P135move | P3373sib | P551resi | P974trib | P921main | P3450spo | P137oper | P937work | P178deve | P26spous | P1303ins | P1435her | P527has  | P155foll | P27count | P58scree | P57direc | P403mout | P306oper | P175perf | P102memb | P1411nom | P40child | P105taxo | P460said | P176manu | P641spor | P22fathe | P31insta | P1344par | P407lang | P1877aft | P159head |"
relations = re.findall(r'P\d+', head_line)
color_schema = ['#1f77b4', '#2ca02c', 'red', 'blue']
markers = ['.', 'x', '*', 'x']
# sizes = [7, 7, 1, 1]
fig = plt.figure(figsize=(3.7,3))


def plot_file(file, fig, c, d, relation1, relation2, f1_a, f1_b):
    # print(relation1, relation2)
    sizes = [5, 5, 1, 1]

    datas = torch.load(file)
    all_labels_tmp = datas['label']
    relation2ids = {}
    is_memorys_tmp = datas['is_memory']
    classifier = datas['classifier']
    relations2logits = {}
    raws_tmp = torch.stack(datas['rep'], 0)
    raws = []
    all_labels = []
    is_memorys = []
    for raw, label, mem in zip(raws_tmp, all_labels_tmp, is_memorys_tmp):
        if label in [relation1, relation2] and not mem:
            raws.append(raw)
            all_labels.append(label)
            is_memorys.append(mem)
    raws = torch.stack(raws, 0)

      
    relation1_right = 0
    relation1_predict = 0
    relation2_right = 0
    relation2_predict = 0

    for relation in [relation1, relation2]:
        logits = raws @ classifier[relation].unsqueeze(1)
        logits = logits.squeeze()
        relations2logits[relation] =  logits


    predict2ids= {f'{relation1}':[], f'{relation2}':[]}
    # for idx, (label, is_memory) in enumerate(zip(all_labels, is_memorys)):
    for idx, (logit1, logit2, label, is_memory) in enumerate(zip(relations2logits[relation1], relations2logits[relation2], all_labels, is_memorys)):

        if is_memory:
            continue

        predict2ids[f'{label}'].append(idx)
    
        if logit1 > logit2:
            relation1_predict += 1
            if label == relation1:
                relation1_right += 1
        else:
            relation2_predict += 1
            if label  == relation2:
                relation2_right += 1


    raws = [x.tolist() for x in raws]
    
    if not os.path.exists(f'{file}_{relation1}_{relation2}-{N_iter}.pt'):
        hiddens = np.array(raws)
        tsne = TSNE(n_components=2, verbose=1, n_iter=N_iter)
        out = tsne.fit_transform(hiddens)
        print('save to file')
        torch.save(out, f'{file}_{relation1}_{relation2}-{N_iter}.pt')
    else:
        out = torch.load(f'{file}_{relation1}_{relation2}-{N_iter}.pt')
        out = np.array(out)

    plt.subplot(2,2,fig)
    relation1_p1 = relation1_right / (relation1_predict + 1e-5)
    relation1_r1 = relation1_right / 140
    relation1_f1 = 2*relation1_p1*relation1_r1 / (relation1_r1 + relation1_p1 + 1e-5)

    relation2_p1 = relation2_right / (relation2_predict + 1e-5)
    relation2_r1 = relation2_right / 140
    relation2_f1 = 2*relation2_p1*relation2_r1 / (relation2_r1 + relation2_p1 + 1e-5)

    f1 = [f1_a, f1_b]
    for i, relation in enumerate(predict2ids):
        idxs = predict2ids[relation]
        reps = [out[x] for x in idxs]
        reps = np.array(reps)
        if len(reps) == 0:
            continue
        
        # if fig <= 2:
        # if f1[i] is not None:
        plt.scatter(c*reps[:, 0], d*reps[:, 1], s=sizes[i], c=color_schema[i], label=f"{relation} (F1: {f1[i]})", marker=markers[i], linewidths=0.5)
        # else:
            # plt.scatter(c*reps[:, 0], d*reps[:, 1], s=sizes[i], c=color_schema[i], label=f"{relation}", marker=markers[i])

        # else:
            # plt.scatter(c*reps[:, 0], d*reps[:, 1], s=sizes[i], c=color_schema[i], label=f"{relation} (F1:{f1[i%2]:.2f})", marker=markers[i])


    plt.xticks([])
    plt.yticks([])
    ax=plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(0.2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(0.2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(0.2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(0.2);####设置上部坐标轴的粗细
    plt.legend(fontsize=5)


rel2f1s={
    ('P25', 'P26'): [0.97, None, 0.97, None, 0.83, 0.73, 0.93, 0.82], 
    ('P156', 'P155'): [0.98, None, 0.96, None, 0.38, 0.49, 0.86, 0.84],
    ('P974', 'P403'):[0.93, None, 0.96, None, 0.82 , 0.81, 0.91, 0.87],
    ('P276', 'P159'):[0.99, None, 0.99, None, 0.63, 0.66, 0.73, 0.71],
    ('P361', 'P31'): [0.66, None, 0.79, None, 0.34, 0.62, 0.53, 0.78],
    ('P26', 'P40'): [0.75, None, 0.76, None, 0.77, 0.74, 0.77 , 0.84],
    ('P3373', 'P22'):[0.88, None, 0.92, None, 0.68, 0.74, 0.76,0.86]

}

if __name__ == "__main__":

    relation1=sys.argv[1]
    relation2=sys.argv[2]
    task_id1 = relations.index(relation1) // 8
    task_id2 = relations.index(relation2) // 8
    print(task_id1)
    print(task_id2)
    f1s = rel2f1s.get((relation1, relation2), [None, None, None, None, None, None, None, None])

    plot_file(f"EMAR-FewRel-M_10/0/{task_id1}.pt", 1, -1, 1, relation1, relation2, f1s[0], f1s[1])
    plot_file(f"EMAR-aca-FewRel-M_10/0/{task_id1}.pt", 2, 1, 1, relation1, relation2, f1s[2], f1s[3])

    plot_file(f"EMAR-FewRel-M_10/0/{task_id2}.pt", 3, 1, -1, relation1, relation2 , f1s[4], f1s[5])
    plot_file(f"EMAR-aca-FewRel-M_10/0/{task_id2}.pt", 4, -1, 1, relation1, relation2, f1s[6], f1s[7])


    fig.tight_layout()
    plt.subplots_adjust(wspace =0.03, hspace =0.03)#调整子图间距
    plt.show()

    plt.savefig(f'{relation1}-{relation2}.png', dpi=400)