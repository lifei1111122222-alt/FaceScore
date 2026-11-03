import math
import torch


@torch.no_grad()
def rating_metrics(pred_probs, target_probs):
    """
    pred_probs:   (B, 10)，预测概率
    target_probs: (B, 10)，真实概率
    每行概率总和应为 1
    """
    pred_probs = pred_probs.float()
    target_probs = target_probs.to(
        device=pred_probs.device,
        dtype=torch.float32,
    )

    # 1. EMD：与之前 EMDLoss 的定义一致
    cdf_diff = (
        pred_probs.cumsum(dim=1)
        - target_probs.cumsum(dim=1)
    )
    emd_per_image = (
        torch.linalg.vector_norm(cdf_diff, ord=2, dim=1)
        / math.sqrt(10)
    )

    # 2. 平均评分 MAE
    scores = torch.arange(
        1, 11,
        device=pred_probs.device,
        dtype=torch.float32,
    )

    pred_mean = (pred_probs * scores).sum(dim=1)
    target_mean = (target_probs * scores).sum(dim=1)
    mae_per_image = (pred_mean - target_mean).abs()

    return {
        "emd": emd_per_image.mean().item(),
        "mean_score_mae": mae_per_image.mean().item(),
        "emd_per_image": emd_per_image,
        "mae_per_image": mae_per_image,
    }