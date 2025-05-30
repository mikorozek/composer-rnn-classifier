import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ComposerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_lstm_layers=2,
        hidden_dim=256,
        fc_dims=[128, 64],
        num_classes=5,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0,
        )

        self.fc_layers = []
        input_dim = hidden_dim

        for fc_dim in fc_dims:
            self.fc_layers.extend(
                [nn.Linear(input_dim, fc_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            input_dim = fc_dim

        self.fc_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*self.fc_layers)

    def forward(self, x, x_lens):
        embedded = self.embedding(x)

        embedded_packed = pack_padded_sequence(embedded,
                                               x_lens,
                                               batch_first=True,
                                               enforce_sorted=False)

        lstm_out, (hidden, cell) = self.lstm(embedded_packed)

        output = self.classifier(hidden[-1])

        return output
