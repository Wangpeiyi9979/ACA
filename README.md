# Introduction
Source code for the long paper [Learning Robust Representations for Continual  Relation Extraction via Adversarial Class Augmentation](https://arxiv.org/abs/2210.04497) in EMNLP 2022. In this paper:
- We conduct a series of empirical studies on two strong CRE methods (RP-CRE and EMAR) and observe that catastrophic forgetting is strongly related with the existence of analogous relations.
- We find an important reason for catastrophic forgetting in CRE which is overlooked in all previous work: the CRE models suffer from learning shortcuts to identify new relations, which are not robust enough against the appearance of their analogous relations.
- We propose an adversarial class augmentation (ACA) mechanism to help CRE models learn more robust representations. Experimental results on two benchmarks show that our method can consistently improve the performance of two state-of-the-art methods.

# Environment
- Python: 3.7.11
- Torch: 1.8.0+cu111
```
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


# Dataset
We use two datasets in our experiments, `FewRel` and `TACRED`:
- FewRel: `data/data_with_marker.json`
- TACRED: `data/data_with_marker_tacred.json`

We construct the $5$ CRE task sequences in both FewRel and TACRED the same as [RP-CRE](https://aclanthology.org/2021.acl-long.20/) and [CRL](https://aclanthology.org/2022.findings-acl.268/).



# Backbone CRE Models
Because our proposed ACA is orthogonal to previous work, we adopt ACA to $2$ strong CRE baselines,  [EMAR](https://aclanthology.org/2020.acl-main.573/) and [RP-CRE](https://aclanthology.org/2021.acl-long.20/).

# Run
```
bash bash/[dataset]/[model].sh [gpu id]
    - dataset: the dataset name, e.g.,:
        - fewrel/tacred
    - model: the model name, e.g.,:
        - emar/rp_cre (the vanilla models)
        - emar_aca/rp_cre_aca (the vanilla models with our ACA)
    - gpu id: the single gpu id, e,g, 0
```

For example, 
```
bash bash/fewrel/emar 0
bash bash/fewrel/emar_aca 0
```


# Citation
```
@article{wang2022learning,
  title={Learning Robust Representations for Continual Relation Extraction via Adversarial Class Augmentation},
  author={Wang, Peiyi and Song, Yifan and Liu, Tianyu and Lin, Binghuai and Cao, Yunbo and Li, Sujian and Sui, Zhifang},
  booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
  year = "2022",
  publisher = "Association for Computational Linguistics",
}
```
