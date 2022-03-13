import torch
import torch.nn as nn

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


class SarcasmDetectionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super(SarcasmDetectionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                            bidirectional=True, num_layers=num_layers, dropout=dropout)
        # self.linear = nn.Linear(2 * hidden_dim, 1)
        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_dim, 36),
            nn.ReLU(),
            nn.Linear(36, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        text = torch.permute(text, (1, 0, 2))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            text, text_lengths, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.linear(hidden)
