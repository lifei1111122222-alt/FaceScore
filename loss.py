import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class TreeOrdinalLoss(nn.Module):
    def __init__(self, nll_weight=1.0, emd_weight=2.0, mean_weight=0.5, rdrop_weight=0.1):
        super().__init__()
        self.nll_weight = nll_weight
        self.emd_weight = emd_weight
        self.mean_weight = mean_weight
        self.rdrop_weight = rdrop_weight
        self.register_buffer("bins", torch.arange(1, 11, dtype=torch.float32).view(1, 10))
    def _single(self, logits, target):
        log_probs = F.log_softmax(logits.float(), dim=1)
        probs = log_probs.exp()
        nll = -(target * log_probs).sum(dim=1)
        emd = (probs.cumsum(dim=1) - target.cumsum(dim=1)).square().mean(dim=1).clamp_min(1e-12).sqrt()
        pred_mean = (probs * self.bins).sum(dim=1)
        true_mean = (target * self.bins).sum(dim=1)
        huber = F.smooth_l1_loss(pred_mean, true_mean, beta=1.0, reduction="none") / 9.0
        return self.nll_weight * nll + self.emd_weight * emd + self.mean_weight * huber, log_probs
    def forward(self, logits, target_probs):
        outputs = logits if isinstance(logits, (tuple, list)) else (logits,)
        if len(outputs) not in (1, 2) or any(t.ndim != 2 or t.shape[1] != 10 for t in outputs):
            raise ValueError("logits 必须是 (B,10) 或两个 (B,10) 的元组")
        target = target_probs.to(device=outputs[0].device, dtype=torch.float32)
        if target.shape != outputs[0].shape or not torch.isfinite(target).all() or (target < 0).any() or not torch.allclose(target.sum(dim=1), torch.ones_like(target[:, 0]), atol=1e-5):
            raise ValueError("target_probs 必须是合法的 (B,10) 概率分布")
        loss_a, logp_a = self._single(outputs[0], target)
        if len(outputs) == 1:
            return loss_a.mean()
        if outputs[1].shape != outputs[0].shape:
            raise ValueError("两次 logits 形状不同")
        loss_b, logp_b = self._single(outputs[1], target)
        kl_ab = (logp_a.exp() * (logp_a - logp_b)).sum(dim=1)
        kl_ba = (logp_b.exp() * (logp_b - logp_a)).sum(dim=1)
        return 0.5 * (loss_a + loss_b).mean() + 0.5 * self.rdrop_weight * (kl_ab + kl_ba).mean()