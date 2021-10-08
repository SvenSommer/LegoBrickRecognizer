import torch
from torch import nn


class ArcFaceLayer(nn.Module):
    def __init__(self,
                 embedding_size=512,
                 num_classes=1000,
                 scale=64.0,
                 margin=0.5,
                 **params):
        """
        Class constructor
        Args:
            embedding_size: layer embedding size
            num_classes: count of classes
        """
        super(ArcFaceLayer, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.W = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes))

    def forward(self, x, y):
        x = nn.functional.normalize(
            x,
            p=2,
            dim=1
        )

        w = nn.functional.normalize(
            self.W,
            p=2,
            dim=0
        )

        y = torch.nn.functional.one_hot(
            y,
            num_classes=self.num_classes
        ).to(x.dtype)
        y = y.view(-1, self.num_classes)

        logits = x @ w  # dot product

        # clip logits to prevent zero division when backward
        theta = torch.acos(logits.clamp(-1.0 + 1E-7, 1.0 - 1E-7))

        target_logits = torch.cos(theta + self.margin)

        logits = logits * (1 - y) + target_logits * y
        logits *= self.scale  # feature re-scale

        return logits
