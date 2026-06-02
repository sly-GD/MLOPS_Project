import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size=6400, embed_dim=256, num_classes=6):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", sparse=False)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, offsets):
        x = self.embedding(input_ids, offsets)
        return self.classifier(x)
