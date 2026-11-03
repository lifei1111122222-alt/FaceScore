"""Recent open-source visual encoder baselines (2025-2026), adapted to 10-bin attractiveness distributions.
All backbones come from timm>=1.0.30 and retain their released pretrained architecture and weights.
Only the image-to-10-bin downstream prediction head and loss are task adaptations.
Input: RGB/RGB+mask float tensor [B,3/4,H,W] in [0,1]. Output: log probabilities [B,10].
"""
import torch
from torch import nn
from torch.nn import functional as F
PAPERS = {
    "EUPE": {"year": 2026, "venue": "arXiv 2026", "paper": "Efficient Universal Perception Encoder", "url": "https://arxiv.org/abs/2603.22387", "model": "vit_small_patch16_dinov3_qkvb.eupe_lvd1689m", "checkpoint": "https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.eupe_lvd1689m", "size": 256, "pool": "token"},
    "TIPSV2": {"year": 2026, "venue": "CVPR 2026", "paper": "TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment", "url": "https://arxiv.org/abs/2604.12012", "model": "vit_base_patch14_reg1_tipsv2.webli", "checkpoint": "https://huggingface.co/timm/vit_base_patch14_reg1_tipsv2.webli", "size": 448},
    "QWEN35": {"year": 2026, "venue": "Qwen3.5 release 2026", "paper": "Qwen3.5: Towards Native Multimodal Agents (visual tower)", "url": "https://qwen.ai/blog?id=qwen3.5", "model": "qwen3_vit_88m.qwen3_5_0_8b", "checkpoint": "https://huggingface.co/timm/qwen3_vit_88m.qwen3_5_0_8b", "size": 256},
    "MOBILECLIP2": {"year": 2025, "venue": "TMLR 2025", "paper": "MobileCLIP2: Improving Multi-Modal Reinforced Training", "url": "https://arxiv.org/abs/2508.20691", "model": "fastvit_mci0.apple_mclip2_dfndr2b", "checkpoint": "https://huggingface.co/timm/fastvit_mci0.apple_mclip2_dfndr2b", "size": 256},
    "PECORE": {"year": 2025, "venue": "2025 research release", "paper": "Perception Encoder (PE-Core)", "url": "https://github.com/facebookresearch/perception_models", "model": "vit_pe_core_small_patch16_384.fb", "checkpoint": "https://huggingface.co/timm/vit_pe_core_small_patch16_384.fb", "size": 384},
    "IFORMER": {"year": 2025, "venue": "ICLR 2025", "paper": "iFormer: Integrating ConvNet and Transformer for Mobile Application", "url": "https://huggingface.co/docs/timm/main/models/iformer", "model": "iformer_m.in1k", "checkpoint": "https://huggingface.co/timm/iformer_m.in1k", "size": 224},
}
METHODS = tuple(PAPERS.keys())
def ensure_timm():
    try:
        import timm
        from packaging.version import Version
    except ImportError as exc:
        raise RuntimeError("请先安装：python -m pip install -U 'timm>=1.0.30' huggingface_hub safetensors") from exc
    if Version(timm.__version__) < Version("1.0.30"):
        raise RuntimeError(f"当前 timm={timm.__version__}，需要 >=1.0.30：python -m pip install -U timm")
    return timm
class PaperVisionBaseline(nn.Module):
    def __init__(self, method, pretrained=True, freeze_backbone=True, dropout=0.35, input_size=None):
        super().__init__()
        key = method.upper().replace("-", "").replace("_", "")
        if key not in PAPERS:
            raise ValueError(f"未知方法 {method}，可选：{METHODS}")
        timm = ensure_timm()
        self.method = key
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        cfg = PAPERS[key]
        model_kwargs = {"num_classes": 0}
        if "pool" in cfg:
            model_kwargs["global_pool"] = cfg["pool"]
        try:
            self.backbone = timm.create_model(cfg["model"], pretrained=pretrained, **model_kwargs)
        except Exception as exc:
            raise RuntimeError(f"{key} 官方架构或预训练权重加载失败：{cfg['model']}。检查 timm 版本、网络访问及 Hugging Face 缓存。不会自动回退到随机权重。") from exc
        data_cfg = timm.data.resolve_model_data_config(self.backbone)
        self.input_size = int(input_size or cfg["size"])
        self.interpolation = "bicubic" if "bicubic" in str(data_cfg.get("interpolation", "bicubic")).lower() else "bilinear"
        self.register_buffer("mean", torch.tensor(data_cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(data_cfg.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32).view(1, 3, 1, 1))
        dim = getattr(self.backbone, "num_features", None)
        if not isinstance(dim, int) or dim < 1:
            raise RuntimeError(f"{key} 骨干没有有效 num_features，无法安全添加预测头")
        self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(dim, 10))
        self.set_frozen(freeze_backbone)
    def set_frozen(self, freeze=True):
        self.freeze_backbone = bool(freeze)
        for param in self.backbone.parameters():
            param.requires_grad_(not freeze)
        if freeze:
            self.backbone.eval()
        return self
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
    def preprocess(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] not in (3, 4):
            raise ValueError("x 必须为 [B,3/4,H,W] 的张量，范围 [0,1]")
        image = x[:, :3].float()
        if image.shape[-2:] != (self.input_size, self.input_size):
            image = F.interpolate(image, size=(self.input_size, self.input_size), mode=self.interpolation, align_corners=False, antialias=True)
        return (image - self.mean) / self.std
    def extract_features(self, x):
        pixels = self.preprocess(x)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            features = self.backbone(pixels)
        if isinstance(features, (tuple, list)):
            raise RuntimeError(f"{self.method} 返回多路特征，不能未经验证地取其一作为原模型特征")
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != self.head[-1].in_features:
            raise RuntimeError(f"{self.method} 特征形状异常：{getattr(features, 'shape', type(features))}")
        return features
    def forward_from_features(self, features, return_pair=False):
        out1 = F.log_softmax(self.head(features).float(), dim=1)
        if return_pair:
            out2 = F.log_softmax(self.head(features).float(), dim=1)
            return out1, out2
        return out1
    def forward(self, x, return_pair=False):
        return self.forward_from_features(self.extract_features(x), return_pair=return_pair)
    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return self.forward(x).exp()
def create_baseline(method, pretrained=True, freeze_backbone=True, dropout=0.35, input_size=None):
    return PaperVisionBaseline(method, pretrained=pretrained, freeze_backbone=freeze_backbone, dropout=dropout, input_size=input_size)
class DistributionLoss(nn.Module):
    def __init__(self,prior=None,alpha=0.5,dm_weight=0.5,emd_weight=1.0,mean_weight=1.0,rank_weight=0.1,rank_temperature=0.5,rdrop_weight=0.0,ce_weight=0.75,nll_weight=None,count_weight_power=0.5,count_weight_max=2.5,tail_weight=0.35,tail_margin=0.02,target_smoothing=2.5,entropy_weight=0.05,dm_per_rating=True,spread_weight=0.0,field_smooth_weight=0.0):
        super().__init__()
        if prior is None:
            prior=torch.full((10,),0.1,dtype=torch.float32)
        prior=torch.as_tensor(prior,dtype=torch.float32)
        if prior.shape!=(10,) or (prior<0).any() or prior.sum()<=0:
            raise ValueError("prior必须是非负的10维概率向量")
        self.alpha=float(alpha)
        self.dm_weight=float(dm_weight if nll_weight is None else nll_weight)
        self.emd_weight=float(emd_weight)
        self.mean_weight=float(mean_weight)
        self.rank_weight=float(rank_weight)
        self.rank_temperature=float(rank_temperature)
        self.rdrop_weight=float(rdrop_weight)
        self.ce_weight=float(ce_weight)
        self.count_weight_power=float(count_weight_power)
        self.count_weight_max=float(count_weight_max)
        self.tail_weight=float(tail_weight)
        self.tail_margin=float(tail_margin)
        self.target_smoothing=float(target_smoothing)
        self.entropy_weight=float(entropy_weight)
        self.spread_weight=float(spread_weight)
        self.field_smooth_weight=float(field_smooth_weight)
        self.dm_per_rating=bool(dm_per_rating)
        if self.count_weight_power<0:
            raise ValueError("count_weight_power必须非负")
        if self.count_weight_max<1:
            raise ValueError("count_weight_max必须至少为1")
        if self.target_smoothing<0:
            raise ValueError("target_smoothing必须非负")
        if self.entropy_weight<0:
            raise ValueError("entropy_weight必须非负")
        if self.spread_weight<0:
            raise ValueError("spread_weight必须非负")
        if self.field_smooth_weight<0:
            raise ValueError("field_smooth_weight必须非负")
        self.register_buffer("prior",prior/prior.sum())
        self.register_buffer("bins",torch.arange(1,11,dtype=torch.float32).view(1,10))
    def _rank(self,pred_mean,true_mean,counts,reliability=None):
        valid=counts.float()>=2
        if int(valid.sum())<2:
            return pred_mean.sum()*0.0
        target_diff=true_mean[:,None]-true_mean[None,:]
        pair=valid[:,None]&valid[None,:]&(target_diff.abs()>=0.35)
        pair=torch.triu(pair,diagonal=1)
        if not bool(pair.any()):
            return pred_mean.sum()*0.0
        pred_diff=pred_mean[:,None]-pred_mean[None,:]
        labels=(target_diff[pair]>0).float()
        pair_loss=F.binary_cross_entropy_with_logits(pred_diff[pair]/self.rank_temperature,labels,reduction='none')
        if reliability is None:
            return pair_loss.mean()
        pair_reliability=((reliability[:,None]+reliability[None,:])*0.5)[pair]
        return (pair_loss*pair_reliability).sum()/pair_reliability.sum().clamp_min(1e-8)
    def _one(self,logp,target,counts,concentration):
        alpha=concentration.float().clamp_min(1e-6)
        counts=counts.float().clamp_min(1.0)
        target=target.float().clamp_min(0.0)
        target=target/target.sum(dim=1,keepdim=True).clamp_min(1e-8)
        observed_target=target
        observed_mean=(observed_target*self.bins).sum(dim=1)
        count_vector=observed_target*counts[:,None]
        total_alpha=alpha.sum(dim=1)
        dm_nll=-(torch.lgamma(total_alpha)-torch.lgamma(total_alpha+counts)+torch.lgamma(alpha+count_vector).sum(dim=1)-torch.lgamma(alpha).sum(dim=1))
        if self.dm_per_rating:
            dm_nll=dm_nll/counts.clamp_min(1.0)
        probs=logp.float().exp().clamp_min(1e-8)
        probs=probs/probs.sum(dim=1,keepdim=True).clamp_min(1e-8)
        prior_batch=self.prior.to(device=target.device,dtype=target.dtype).view(1,10)
        calibration_target=observed_target
        if self.target_smoothing>0:
            calibration_target=(observed_target*counts[:,None]+self.target_smoothing*prior_batch)/(counts[:,None]+self.target_smoothing)
            calibration_target=calibration_target/calibration_target.sum(dim=1,keepdim=True).clamp_min(1e-8)
        cdf_diff=probs.cumsum(dim=1)-observed_target.cumsum(dim=1)
        emd=cdf_diff.square().mean(dim=1).clamp_min(1e-8).sqrt()
        pred_mean=(probs*self.bins).sum(dim=1)
        true_mean=(observed_target*self.bins).sum(dim=1)
        mean_loss=F.smooth_l1_loss(pred_mean,true_mean,beta=1.0,reduction='none')/9.0
        cross_entropy=-(calibration_target*logp.float()).sum(dim=1)
        target_tail=observed_target[:,7:].sum(dim=1)
        predicted_tail=probs[:,7:].sum(dim=1)
        tail_relevance=torch.sigmoid((observed_mean-6.75)/0.65)*(counts/(counts+1.0)).sqrt()
        tail_confidence=(counts/(counts+self.target_smoothing).clamp_min(1e-8)).sqrt()
        tail_loss=F.relu(target_tail-predicted_tail-self.tail_margin).square()*tail_relevance*tail_confidence
        predicted_entropy=-(probs*probs.clamp_min(1e-8).log()).sum(dim=1)
        target_entropy=-(calibration_target*calibration_target.clamp_min(1e-8).log()).sum(dim=1)
        entropy_loss=((predicted_entropy-target_entropy.detach())/2.3025851).square()*tail_confidence
        reliability=counts.pow(self.count_weight_power).clamp(0.8,self.count_weight_max)
        rank=self._rank(pred_mean,observed_mean,counts,reliability)
        reliable_mask=counts>=2.0
        if int(reliable_mask.sum())>3:
            reliable_mean=observed_mean[reliable_mask]
            reliable_target=observed_target[reliable_mask]
            reliable_counts=counts[reliable_mask]
            observed_variance=(reliable_target*(self.bins-reliable_mean[:,None]).square()).sum(dim=1)
            signal_variance=((reliable_mean-reliable_mean.mean()).square()-observed_variance/reliable_counts.clamp_min(1.0)).clamp_min(0.0).mean().detach()
            predicted_variance=(pred_mean[reliable_mask]-pred_mean[reliable_mask].mean()).square().mean()
            spread_loss=((predicted_variance-signal_variance)/2.0).square()
        else:
            spread_loss=pred_mean.sum()*0.0
        log_centered=logp.float()-logp.float().mean(dim=1,keepdim=True)
        force=-log_centered[:,1:]+log_centered[:,:-1]
        force_curvature=force[:,1:]-force[:,:-1]
        field_smooth=force_curvature.square().mean(dim=1)
        point=self.dm_weight*dm_nll+self.ce_weight*cross_entropy+self.emd_weight*emd+self.mean_weight*mean_loss+self.tail_weight*tail_loss+self.entropy_weight*entropy_loss+self.spread_weight*spread_loss+self.field_smooth_weight*field_smooth
        weighted_point=(point*reliability).sum()/reliability.sum().clamp_min(1e-8)
        return weighted_point + self.rank_weight * rank,probs
    def forward(self,output,target,counts=None,concentration=None):
        outputs=output if isinstance(output,(tuple,list)) else (output,)
        concentrations=concentration if isinstance(concentration,(tuple,list)) else (concentration,)
        if len(outputs) not in (1,2):
            raise ValueError("输出必须是一个或两个[B,10]张量")
        if concentration is None:
            raise ValueError("ECDT训练必须传入模型返回的Dirichlet浓度参数")
        if len(concentrations)!=len(outputs):
            raise ValueError("输出与Dirichlet浓度数量不一致")
        target=target.to(device=outputs[0].device,dtype=torch.float32)
        if counts is None:
            counts=torch.ones(target.shape[0],device=outputs[0].device,dtype=torch.float32)
        counts=counts.to(device=outputs[0].device,dtype=torch.float32)
        loss_a,prob_a=self._one(outputs[0],target,counts,concentrations[0])
        if len(outputs)==1:
            return loss_a
        loss_b,prob_b=self._one(outputs[1],target,counts,concentrations[1])
        log_a=prob_a.clamp_min(1e-8).log()
        log_b=prob_b.clamp_min(1e-8).log()
        kl_ab=(prob_a*(log_a-log_b)).sum(dim=1).mean()
        kl_ba=(prob_b*(log_b-log_a)).sum(dim=1).mean()
        return 0.5*(loss_a+loss_b)+0.5*self.rdrop_weight*(kl_ab+kl_ba)
