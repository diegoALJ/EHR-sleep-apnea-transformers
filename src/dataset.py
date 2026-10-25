import torch
from torch.utils.data import Dataset
from .temporal_processing import tokenize_recent_history


class PatientDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.labels = frame['label'].tolist()
        self.visits = frame['visits'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tok = tokenize_recent_history(self.visits[index], self.tokenizer, self.max_length)
        return {**{k: torch.tensor(v, dtype=torch.long) for k, v in tok.items()},
                'labels': torch.tensor(self.labels[index], dtype=torch.long)}
