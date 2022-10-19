import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import random


class ReDataset(Dataset):

    def __init__(self, data, config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        idxs = [item.get('idx', 0) for item in data] 
        label = torch.tensor([item['relation'] for item in data])
        tokens = [torch.tensor(item['tokens']) for item in data]
        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        strings = [item.get('string', 'None') for item in data] 
        is_pre_data = [item.get('is_pre_data', False) for item in data] 
        return (
            idxs, # not use
            label,
            tokens,
            strings, # not use
            is_pre_data # not use
        )

def get_data_loader(config, data, shuffle=False, drop_last=False, batch_size=None, sampler=None):
    dataset = ReDataset(data, config)

    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader