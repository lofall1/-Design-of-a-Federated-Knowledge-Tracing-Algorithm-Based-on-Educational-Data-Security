import torch
import torch.nn as nn

class DKT(nn.Module):
    def __init__(self, num_questions, embed_dim=64, hidden_dim=128):
        super(DKT, self).__init__()

        self.embedding = nn.Embedding(num_questions, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_questions)

    def forward(self, q, a):
        q_embed = self.embedding(q)
        a = a.unsqueeze(-1)
        x = torch.cat([q_embed, a], dim=-1)

        out, _ = self.lstm(x)
        logits = self.fc(out)

        return torch.sigmoid(logits)