import pickle

import torch
from torch.utils.data import Dataset


class ComposerDataset(Dataset):

    def __init__(self, data_path, is_test=False):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.is_test = is_test

        if not is_test:
            self.sequences, self.labels = zip(*self.data)
        else:
            self.sequences = self.data
            self.labels = None

        self.vocab_size = self._find_vocab_size()

    def _find_vocab_size(self):
        max_idx = 0

        for sequence in self.sequences:
            if len(sequence) > 0:
                max_idx = max(max_idx, max(sequence))

        return int(max_idx + 3)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        sequence = [int(token) + 2 for token in sequence]

        sequence_tensor = torch.tensor(sequence, dtype=torch.long)

        if not self.is_test:
            label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return sequence_tensor, label
        else:
            return sequence_tensor
