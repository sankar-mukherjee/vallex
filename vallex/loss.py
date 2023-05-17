import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
import torch

NUM_TEXT_TOKENS = 128
NUM_AUDIO_TOKENS = 1024  # EnCodec RVQ bins

class compute_loss(nn.Module):
    def __init__(self, device):
        super(compute_loss, self).__init__()

        self.reduction="sum"
        self.ar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        ).to(device)
        self.nar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        ).to(device)

    def forward(self, y_lens, ar_logits, ar_targets, nar_logits, nar_targets):

        total_loss, metrics = 0.0, {}
        
        # AR loss
        total_loss = F.cross_entropy(
            ar_logits, 
            ar_targets, 
            reduction=self.reduction
        )
        metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(
        ar_logits.detach(), ar_targets
        ).item() * y_lens.sum().type(torch.float32)

        # NAR loss
        total_loss += F.cross_entropy(
            nar_logits,
            nar_targets,
            ignore_index=NUM_AUDIO_TOKENS,
            reduction=self.reduction,
        )
        metrics["NarTop10Accuracy"] = (
            self.nar_accuracy_metric(
                F.pad(
                    nar_logits.detach(),
                    (0, 0, 0, 1, 0, 0),
                    value=nar_logits.min().cpu().item(),
                ),
                nar_targets,
            ).item() * y_lens.sum().type(torch.float32)
        )

        return total_loss / 2.0, metrics
