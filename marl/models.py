import torch
from torch import nn


class RecurrentPolicy(nn.Module):
    def __init__(self, dual=False, hidden=128, observation_size=13, n_actions=4):
        super().__init__()
        self.dual, self.hidden_size = dual, hidden
        self.observation_size, self.n_actions = observation_size, n_actions
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        heads = 2 if dual else 1  # 0: target/single, 1: exploration
        self.actors = nn.ModuleList([nn.Linear(hidden, n_actions) for _ in range(heads)])
        self.critics = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(heads)])

    def forward(self, observations, hidden=None):
        features, hidden = self.gru(self.encoder(observations), hidden)
        logits = torch.stack([actor(features) for actor in self.actors], dim=-2)
        values = torch.cat([critic(features) for critic in self.critics], dim=-1)
        return logits, values, hidden

    def save(self, path, **metadata):
        torch.save(dict(state_dict=self.state_dict(), dual=self.dual,
                        hidden=self.hidden_size, observation_size=self.observation_size,
                        n_actions=self.n_actions, metadata=metadata), path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(checkpoint['dual'], checkpoint['hidden'],
                    checkpoint.get('observation_size', 13), checkpoint.get('n_actions', 4))
        model.load_state_dict(checkpoint['state_dict'])
        return model, checkpoint['metadata']
