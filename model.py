import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def feature_entropy(nodes, temperature=0.35):
    channels = nodes.shape[-1]
    logits = F.layer_norm(nodes.float(), (channels,)) / temperature
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True) / math.log(channels)
def grid_edge_index(side):
    indices = torch.arange(side * side).reshape(side, side)
    src = torch.cat([indices[:, :-1].reshape(-1), indices[:-1, :].reshape(-1)])
    dst = torch.cat([indices[:, 1:].reshape(-1), indices[1:, :].reshape(-1)])
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
class ConnectedTreePool(nn.Module):
    def __init__(self, dim, group_ids, n_groups, use_entropy=True):
        super().__init__()
        group_ids = torch.as_tensor(group_ids, dtype=torch.long).reshape(-1)
        if group_ids.numel() % n_groups != 0 or torch.bincount(group_ids, minlength=n_groups).min().item() != group_ids.numel() // n_groups:
            raise ValueError("每个聚类必须拥有相同且非零的子节点数量")
        self.n_groups = n_groups
        self.use_entropy = bool(use_entropy)
        self.child_log_count = math.log(group_ids.numel() // n_groups)
        self.register_buffer("group_ids", group_ids)
        self.register_buffer("membership", F.one_hot(group_ids, num_classes=n_groups).bool())
        self.gate = nn.Sequential(nn.LayerNorm(dim + 1), nn.Linear(dim + 1, 1))
        self.entropy_projection = nn.Sequential(nn.LayerNorm(dim + 1), nn.Linear(dim + 1, dim))
    def forward(self, nodes, child_entropy, mask=None):
        augmented = torch.cat([nodes, child_entropy], dim=-1)
        logits = self.gate(augmented).squeeze(-1).unsqueeze(-1).expand(-1, -1, self.n_groups)
        if mask is not None:
            logits = logits + (mask.clamp(0, 1) + 0.1).log().unsqueeze(-1)
        logits = logits.masked_fill(~self.membership.unsqueeze(0), -1e4)
        attention = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bnk,bnd->bkd", attention, nodes)
        attention_entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=1).unsqueeze(-1) / self.child_log_count
        node_entropy = 0.5 * (attention_entropy + feature_entropy(pooled)) if self.use_entropy else torch.zeros_like(attention_entropy)
        pooled = pooled + 0.1 * self.entropy_projection(torch.cat([pooled, node_entropy], dim=-1))
        return pooled, node_entropy, attention
class RatingNet(nn.Module):
    def __init__(self,
                 pretrained=True,
                 dropout=0.5,
                 train_last_stage=False,
                 width=96,
                 prior_mix=0.05,
                 use_entropy=True,
                 use_high_resolution=True,
                 backbone=None,
                 scales=(224,336,448,560),
                 grid_side=16,
                 use_evidence_head=False,
                 evidence_base=0.5,
                 use_global_token=True):
        super().__init__()
        if not 0 <= prior_mix < 1:
            raise ValueError("prior_mix 必须在 [0,1) 之间")
        if backbone is None and not pretrained:
            raise ValueError("建议使用预训练 DINOv2，或通过 backbone 参数传入已有视觉骨干")
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained, trust_repo=True) if backbone is None else backbone
        self.prior_mix = float(prior_mix)
        self.use_entropy = bool(use_entropy)
        self.use_high_resolution = bool(use_high_resolution)
        self.scales = tuple(scales) if self.use_high_resolution else (224,)
        if not self.scales or self.scales[0] != 224 or len(set(self.scales)) != len(self.scales) or any(not isinstance(scale, int) or scale < 224 or scale % 14 != 0 for scale in self.scales):
            raise ValueError("scales 必须以224开头、不重复，并且每个分辨率均为大于等于224的14的倍数")
        self.scale_specs = ((224, 5), (224, 8), (224, 11)) + tuple((scale, 11) for scale in self.scales if scale != 224)
        if grid_side not in (16, 32):
            raise ValueError("grid_side 当前支持16或32")
        self.grid_side = int(grid_side)
        self.region_side = 4
        self.super_side = 2
        self.width = width
        embed_dim = self.backbone.embed_dim
        self.use_global_token=bool(use_global_token)
        self.global_projection=nn.Sequential(nn.LayerNorm(embed_dim),nn.Linear(embed_dim,width),nn.GELU(),nn.Dropout(0.1)) if self.use_global_token else None
        self.global_gate=nn.Parameter(torch.tensor(-1.5)) if self.use_global_token else None
        self.projection = nn.Sequential(nn.Conv2d(embed_dim, width, 1), nn.GroupNorm(1, width), nn.GELU(), nn.Dropout2d(0.1))
        self.source_scale = nn.Parameter(torch.ones(len(self.scale_specs), width))
        self.source_bias = nn.Parameter(torch.zeros(len(self.scale_specs), width))
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scale_specs)))
        self.spatial_update = nn.Sequential(nn.Conv2d(width * 2, width, 1), nn.GELU(), nn.Dropout2d(0.2), nn.Conv2d(width, width, 1))
        self.region_update = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, width))
        self.super_update = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, width))
        yy, xx = torch.meshgrid(torch.arange(self.grid_side), torch.arange(self.grid_side), indexing="ij")
        group_side = self.grid_side // self.region_side
        leaf_parents = (yy // group_side * self.region_side + xx // group_side).reshape(-1)
        ry, rx = torch.meshgrid(torch.arange(self.region_side), torch.arange(self.region_side), indexing="ij")
        region_parents = (ry // 2 * self.super_side + rx // 2).reshape(-1)
        self.pool_leaf = ConnectedTreePool(width, leaf_parents, self.region_side ** 2, use_entropy=use_entropy)
        self.pool_region = ConnectedTreePool(width, region_parents, self.super_side ** 2, use_entropy=use_entropy)
        self.pool_super = ConnectedTreePool(width, torch.zeros(self.super_side ** 2, dtype=torch.long), 1, use_entropy=use_entropy)
        self.use_evidence_head=bool(use_evidence_head)
        self.evidence_base=float(evidence_base)
        if self.evidence_base<=0:
            raise ValueError("evidence_base必须大于0")
        self.head=nn.Sequential(nn.LayerNorm(width*4+4),nn.Linear(width*4+4,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,2))
        if self.use_evidence_head:
            self.node_evidence_head=nn.Sequential(nn.LayerNorm(width+1),nn.Linear(width+1,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,11))
            self.global_evidence_head=nn.Sequential(nn.LayerNorm(width*4+4),nn.Linear(width*4+4,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,11))
            self.entropy_log_scale=nn.Parameter(torch.tensor(-1.0))
            self.local_evidence_logit=nn.Parameter(torch.tensor(-1.5))
            self.evidence_gate=nn.Linear(5,1)
            nn.init.zeros_(self.evidence_gate.weight)
            nn.init.constant_(self.evidence_gate.bias,-1.1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("cuts", torch.arange(1.5, 10.0, 1.0).view(1, 9))
        self.register_buffer("prior_probs", torch.full((10,), 0.1))
        self.register_buffer("neighbor_kernel", torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3))
        self.register_buffer("leaf_edge_index", grid_edge_index(self.grid_side))
        self.register_buffer("region_edge_index", grid_edge_index(self.region_side))
        self.register_buffer("super_edge_index", grid_edge_index(self.super_side))
        self.train_last_stage = False
        self.set_train_last_stage(train_last_stage)
    def set_prior(self, prior):
        prior = torch.as_tensor(prior, dtype=self.prior_probs.dtype, device=self.prior_probs.device)
        if prior.shape != (10,) or not torch.isfinite(prior).all() or (prior < 0).any() or prior.sum() <= 0:
            raise ValueError("prior 必须是非负有限的10维评分频数或概率")
        self.prior_probs.copy_((prior + 1e-3) / (prior.sum() + 1e-2))
        if self.use_evidence_head:
            with torch.no_grad():
                prior_log=self.prior_probs.clamp_min(1e-6).log()
                self.node_evidence_head[-1].bias[:10].copy_(prior_log)
                self.global_evidence_head[-1].bias[:10].copy_(prior_log)
                inverse_softplus=torch.log(torch.expm1(torch.tensor(1.0,dtype=self.prior_probs.dtype,device=self.prior_probs.device)))
                self.node_evidence_head[-1].bias[10].copy_(inverse_softplus)
                self.global_evidence_head[-1].bias[10].copy_(inverse_softplus)
    def set_train_last_stage(self, enabled=False):
        self.train_last_stage = bool(enabled)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if enabled:
            for p in self.backbone.blocks[-1].parameters():
                p.requires_grad_(True)
            for p in self.backbone.norm.parameters():
                p.requires_grad_(True)
        self.backbone.eval()
        return self
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    def _feature_maps(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("x 必须是 RGB 图像张量 (B,3,H,W)")
        token_maps = []
        with torch.set_grad_enabled(self.training and self.train_last_stage):
            for scale in self.scales:
                image = F.interpolate(x, size=(scale, scale), mode="bilinear", align_corners=False, antialias=True)
                image = (image - self.mean) / self.std
                selected_layers = (5, 8, 11) if scale == 224 else (11,)
                layers = self.backbone.get_intermediate_layers(image, n=selected_layers, return_class_token=self.use_global_token and scale==224, norm=True)
                for layer_index,layer_output in enumerate(layers):
                    if isinstance(layer_output,(tuple,list)) and len(layer_output)==2 and torch.is_tensor(layer_output[0]):
                        patch_tokens,global_token=layer_output
                    else:
                        patch_tokens=layer_output
                        global_token=None
                    if self.use_global_token and scale==224 and layer_index==len(selected_layers)-1:
                        selected_global_token=global_token if global_token is not None else patch_tokens.mean(dim=1)
                    token_maps.append((patch_tokens, scale // 14))
        outputs = []
        for i, (patch_tokens, side) in enumerate(token_maps):
            patch_map = patch_tokens.transpose(1, 2).reshape(x.shape[0], self.backbone.embed_dim, side, side)
            if side > self.grid_side:
                patch_map = F.adaptive_avg_pool2d(patch_map, (self.grid_side, self.grid_side))
            if side < self.grid_side:
                patch_map = F.interpolate(patch_map, size=(self.grid_side, self.grid_side), mode="bilinear", align_corners=False)
            projected = self.projection(patch_map)
            projected = projected * self.source_scale[i].view(1, -1, 1, 1) + self.source_bias[i].view(1, -1, 1, 1)
            outputs.append(projected)
        weights = torch.softmax(self.scale_logits, dim=0)
        feature_map=sum(weight * feature for weight, feature in zip(weights, outputs))
        if self.use_global_token:
            if 'selected_global_token' not in locals():
                selected_global_token=feature_map.flatten(2).mean(dim=2)
            global_feature=self.global_projection(selected_global_token.float())
        else:
            global_feature=None
        return feature_map,global_feature
    @torch.no_grad()
    def get_scale_weights(self):
        weights = torch.softmax(self.scale_logits, dim=0).detach().cpu().tolist()
        return {f"{scale}x{scale}_layer{layer + 1}": round(weight, 5) for (scale, layer), weight in zip(self.scale_specs, weights)}
    def _neighbor_mean(self, feature_map):
        channels = feature_map.shape[1]
        kernel = self.neighbor_kernel.to(dtype=feature_map.dtype)
        summed = F.conv2d(feature_map, kernel.expand(channels, 1, 3, 3), padding=1, groups=channels)
        counts = F.conv2d(torch.ones_like(feature_map[:, :1]), kernel, padding=1).clamp_min(1.0)
        return summed / counts
    def _update_nodes(self, nodes, side, layer):
        node_map = nodes.transpose(1, 2).reshape(nodes.shape[0], self.width, side, side)
        neighbors = self._neighbor_mean(node_map).flatten(2).transpose(1, 2)
        return nodes + 0.15 * layer(torch.cat([nodes, neighbors], dim=-1))
    def encode_tree(self, x, face_mask=None):
        feature_map,global_feature = self._feature_maps(x)
        messages = self._neighbor_mean(feature_map)
        feature_map = feature_map + 0.15 * self.spatial_update(torch.cat([feature_map, messages], dim=1))
        leaf_nodes = feature_map.flatten(2).transpose(1, 2)
        leaf_entropy = feature_entropy(leaf_nodes) if self.use_entropy else torch.zeros_like(leaf_nodes[:, :, :1])
        mask_leaf = F.interpolate(face_mask.float(), size=(self.grid_side, self.grid_side), mode="area").flatten(1) if face_mask is not None else None
        region_nodes, region_entropy, leaf_weights = self.pool_leaf(leaf_nodes, leaf_entropy, mask=mask_leaf)
        region_nodes = self._update_nodes(region_nodes, self.region_side, self.region_update)
        super_nodes, super_entropy, region_weights = self.pool_region(region_nodes, region_entropy)
        super_nodes = self._update_nodes(super_nodes, self.super_side, self.super_update)
        root_nodes, root_entropy, super_weights = self.pool_super(super_nodes, super_entropy)
        if self.use_global_token and global_feature is not None:
            root_nodes=root_nodes+torch.sigmoid(self.global_gate)*global_feature.unsqueeze(1)
        summary = torch.cat([leaf_nodes.mean(dim=1), region_nodes.mean(dim=1), super_nodes.mean(dim=1), root_nodes[:, 0], leaf_entropy.mean(dim=1), region_entropy.mean(dim=1), super_entropy.mean(dim=1), root_entropy[:, 0]], dim=1)
        tree = {"leaf_nodes": torch.cat([leaf_nodes, leaf_entropy], dim=-1), "region_nodes": torch.cat([region_nodes, region_entropy], dim=-1), "super_nodes": torch.cat([super_nodes, super_entropy], dim=-1), "root_node": torch.cat([root_nodes, root_entropy], dim=-1), "leaf_to_region": self.pool_leaf.group_ids, "region_to_super": self.pool_region.group_ids, "super_to_root": self.pool_super.group_ids, "leaf_edge_index": self.leaf_edge_index, "region_edge_index": self.region_edge_index, "super_edge_index": self.super_edge_index, "leaf_weights": leaf_weights, "region_weights": region_weights, "super_weights": super_weights}
        return summary, tree
    def _predict_logprobs(self,summary):
        params=self.head(summary).float()
        mu=1.0+9.0*torch.sigmoid(params[:,:1])
        sigma=0.45+2.6*torch.sigmoid(params[:,1:2])
        cdf=torch.sigmoid((self.cuts-mu)/sigma)
        probs=torch.cat([cdf[:,:1],cdf[:,1:]-cdf[:,:-1],1.0-cdf[:,-1:]],dim=1)
        probs=(1.0-self.prior_mix)*probs+self.prior_mix*self.prior_probs.view(1,10)
        return probs.clamp_min(1e-12).log()
    def _node_evidence(self,nodes):
        params=self.node_evidence_head(nodes).float()
        rating_probs=torch.softmax(params[...,:10],dim=-1)
        raw_evidence=params[...,10:11]
        entropy=nodes[...,-1:].clamp(0.0,1.0)
        entropy_decay=torch.exp(-F.softplus(self.entropy_log_scale)*entropy)
        evidence=(F.softplus(raw_evidence)+0.05)*entropy_decay
        return rating_probs,evidence
    def _transport_evidence(self,child_probs,child_evidence,weights,parent_nodes):
        transported_probs=torch.einsum('bnk,bnd->bkd',weights,child_probs)
        transported_evidence=torch.einsum('bnk,bnq->bkq',weights,child_evidence)
        local_probs,local_evidence=self._node_evidence(parent_nodes)
        local_weight=torch.sigmoid(self.local_evidence_logit)
        total_evidence=transported_evidence+local_weight*local_evidence
        total_probs=(transported_evidence*transported_probs+local_weight*local_evidence*local_probs)/total_evidence.clamp_min(1e-6)
        return total_probs,total_evidence
    def _predict_evidence(self,summary,tree):
        leaf_probs,leaf_evidence=self._node_evidence(tree['leaf_nodes'])
        region_probs,region_evidence=self._transport_evidence(leaf_probs,leaf_evidence,tree['leaf_weights'],tree['region_nodes'])
        super_probs,super_evidence=self._transport_evidence(region_probs,region_evidence,tree['region_weights'],tree['super_nodes'])
        root_probs,root_evidence=self._transport_evidence(super_probs,super_evidence,tree['super_weights'],tree['root_node'])
        global_params=self.global_evidence_head(summary).float()
        global_probs=torch.softmax(global_params[:,:10],dim=1)
        global_entropy=summary[:,-4:].mean(dim=1,keepdim=True).clamp(0.0,1.0)
        global_decay=torch.exp(-F.softplus(self.entropy_log_scale)*global_entropy)
        global_evidence=(F.softplus(global_params[:,10:11])+0.05)*global_decay
        global_weight=torch.sigmoid(self.local_evidence_logit)
        base_probs=self._predict_logprobs(summary).exp()
        evidence_e=root_evidence[:,0]+global_weight*global_evidence
        evidence_probs=(root_evidence[:,0]*root_probs[:,0]+global_weight*global_evidence*global_probs)/evidence_e.clamp_min(1e-6)
        entropy_features=torch.cat([tree['root_node'][:,:,-1],summary[:,-4:]],dim=1)
        evidence_gate=torch.sigmoid(self.evidence_gate(entropy_features.float()))
        total_evidence=evidence_e
        total_probs=evidence_gate*evidence_probs+(1.0-evidence_gate)*base_probs
        entropy_level=entropy_features.mean(dim=1,keepdim=True).clamp(0.0,1.0)
        confidence_temperature=0.80+0.35*entropy_level
        total_probs=F.softmax(total_probs.clamp_min(1e-8).log()/confidence_temperature,dim=1)
        alpha=self.evidence_base*self.prior_probs.view(1,10)+total_evidence*total_probs
        alpha=alpha.clamp_min(1e-6)
        log_probs=alpha.log()-alpha.sum(dim=1,keepdim=True).log()
        return log_probs,alpha
    def forward(self,x,return_pair=False,return_tree=False,return_evidence=False):
        if x.ndim!=4 or x.shape[1] not in (3,4):
            raise ValueError("输入必须为 (B,3,H,W) 或 (B,4,H,W)")
        face_mask=x[:,3:4] if x.shape[1]==4 else None
        summary,tree=self.encode_tree(x[:,:3],face_mask=face_mask)
        if self.use_evidence_head:
            log_a,alpha_a=self._predict_evidence(summary,tree)
            if return_pair:
                log_b,alpha_b=self._predict_evidence(summary,tree)
                outputs=(log_a,log_b)
                concentrations=(alpha_a,alpha_b)
            else:
                outputs=log_a
                concentrations=alpha_a
        else:
            outputs=(self._predict_logprobs(summary),self._predict_logprobs(summary)) if return_pair else self._predict_logprobs(summary)
            concentrations=None
        if return_evidence and return_tree:
            return outputs,tree,concentrations
        if return_evidence:
            return outputs,concentrations
        if return_tree:
            return outputs,tree
        return outputs
    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return self.forward(x).softmax(dim=1)

class SparseLandauScoreField(nn.Module):
    def __init__(self,pretrained=True,dropout=0.05,train_last_stage=False,width=32,prior_mix=0.0,use_entropy=False,use_high_resolution=False,backbone=None,scales=(224,),grid_side=16,use_evidence_head=True,evidence_base=2.0,use_global_token=True):
        super().__init__()
        self.backbone=torch.hub.load("facebookresearch/dinov2","dinov2_vits14",pretrained=pretrained,trust_repo=True) if backbone is None else backbone
        self.embed_dim=int(self.backbone.embed_dim)
        self.width=int(width)
        self.prior_mix=float(prior_mix)
        if not 0.0<=self.prior_mix<0.25:
            raise ValueError("prior_mix必须在[0,0.25)之间")
        self.evidence_base=float(evidence_base)
        self.summary_dim=self.embed_dim if bool(getattr(self.backbone,"global_only",False)) else self.embed_dim*9
        self.feature_projection=nn.Sequential(nn.LayerNorm(self.summary_dim),nn.Linear(self.summary_dim,self.width),nn.GELU(),nn.Dropout(float(dropout)))
        self.direct_residual_head=nn.Linear(self.width,10)
        self.force_head=nn.Sequential(nn.LayerNorm(self.width),nn.Linear(self.width,16),nn.GELU(),nn.Linear(16,4))
        self.temperature_head=nn.Linear(self.width,1)
        self.free_energy_head=nn.Sequential(nn.LayerNorm(self.summary_dim),nn.Linear(self.summary_dim,64),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(64,10))
        self.register_buffer("prior_probs",torch.full((10,),0.1,dtype=torch.float32))
        self.output_temperature=1.10
        force_grid=torch.arange(9,dtype=torch.float32).view(-1,1)
        force_modes=torch.arange(4,dtype=torch.float32).view(1,-1)
        force_basis=torch.cos(torch.pi*(force_grid+0.5)*force_modes/9.0)
        force_basis[:,0]=1.0
        force_basis=force_basis/force_basis.norm(dim=0,keepdim=True).clamp_min(1e-6)
        self.register_buffer("force_basis",force_basis)
        self.register_buffer("feature_mean",torch.zeros(self.summary_dim,dtype=torch.float32))
        self.register_buffer("feature_std",torch.ones(self.summary_dim,dtype=torch.float32))
        self.register_buffer("mean",torch.tensor([0.485,0.456,0.406],dtype=torch.float32).view(1,3,1,1))
        self.register_buffer("std",torch.tensor([0.229,0.224,0.225],dtype=torch.float32).view(1,3,1,1))
        nn.init.zeros_(self.direct_residual_head.weight)
        nn.init.zeros_(self.direct_residual_head.bias)
        nn.init.zeros_(self.force_head[-1].weight)
        nn.init.zeros_(self.force_head[-1].bias)
        nn.init.zeros_(self.temperature_head.weight)
        nn.init.constant_(self.temperature_head.bias,-1.2)
        nn.init.zeros_(self.free_energy_head[-1].weight)
        nn.init.zeros_(self.free_energy_head[-1].bias)
        self.set_train_last_stage(train_last_stage)
    def set_prior(self,prior):
        prior=torch.as_tensor(prior,dtype=self.prior_probs.dtype,device=self.prior_probs.device)
        if prior.shape!=(10,) or not torch.isfinite(prior).all() or (prior<0).any() or prior.sum()<=0:
            raise ValueError("prior必须是非负有限的10维评分频数或概率")
        self.prior_probs.copy_((prior+1e-3)/(prior.sum()+1e-2))
    def set_feature_stats(self,mean,std):
        mean=torch.as_tensor(mean,dtype=self.feature_mean.dtype,device=self.feature_mean.device).reshape(-1)
        std=torch.as_tensor(std,dtype=self.feature_std.dtype,device=self.feature_std.device).reshape(-1)
        if mean.shape!=self.feature_mean.shape or std.shape!=self.feature_std.shape:
            raise ValueError(f"特征统计量必须为({self.summary_dim},)")
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-4))
        return self
    def set_train_last_stage(self,enabled=False):
        self.train_last_stage=bool(enabled)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if self.train_last_stage and hasattr(self.backbone,"blocks"):
            for parameter in self.backbone.blocks[-1].parameters():
                parameter.requires_grad_(True)
            if hasattr(self.backbone,"norm"):
                for parameter in self.backbone.norm.parameters():
                    parameter.requires_grad_(True)
        self.backbone.eval()
        return self
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    @torch.no_grad()
    def encode_features(self,x):
        if x.ndim!=4 or x.shape[1] not in (3,4):
            raise ValueError("x必须为(B,3,H,W)或(B,4,H,W)")
        input_size=int(getattr(self.backbone,"input_size",336))
        image=F.interpolate(x[:,:3].float(),size=(input_size,input_size),mode="bilinear",align_corners=False,antialias=True)
        image=(image-self.mean)/self.std
        if bool(getattr(self.backbone,"global_only",False)):
            return self.backbone.encoder(image).float()
        outputs=self.backbone.get_intermediate_layers(image,n=3,return_class_token=True,norm=True)
        if isinstance(outputs,(tuple,list)) and len(outputs)==2 and torch.is_tensor(outputs[0]) and torch.is_tensor(outputs[1]):
            layers=[outputs]
        elif isinstance(outputs,(tuple,list)):
            layers=list(outputs)
        else:
            layers=[outputs]
        global_tokens=[]
        patch_tokens=[]
        for layer in layers:
            if isinstance(layer,(tuple,list)) and len(layer)==2 and torch.is_tensor(layer[0]):
                tokens,global_token=layer
            else:
                tokens=layer
                global_token=tokens.mean(dim=1)
            if global_token is None:
                global_token=tokens.mean(dim=1)
            global_tokens.append(global_token)
            patch_tokens.append(tokens)
        tokens=patch_tokens[-1]
        side=int(tokens.shape[1]**0.5)
        if side*side!=tokens.shape[1]:
            raise RuntimeError("DINOv2 patch token数量不是平方数")
        if x.shape[1]==4:
            mask=F.interpolate(x[:,3:4].float(),size=(side,side),mode="area").flatten(1).clamp_min(0.0)
            denominator=mask.sum(dim=1,keepdim=True)
            uniform=torch.full_like(mask,1.0/mask.shape[1])
            weights=mask/denominator.clamp_min(1e-6)
            weights=torch.where(denominator>1e-6,weights,uniform)
        else:
            weights=torch.full((x.shape[0],tokens.shape[1]),1.0/tokens.shape[1],device=tokens.device,dtype=tokens.dtype)
        patch_mean=(tokens*weights.unsqueeze(-1)).sum(dim=1)
        patch_var=((tokens-patch_mean.unsqueeze(1)).square()*weights.unsqueeze(-1)).sum(dim=1)
        token_grid=tokens.view(tokens.shape[0],side,side,tokens.shape[-1])
        weight_grid=weights.view(weights.shape[0],side,side)
        quadrant_means=[]
        half=side//2
        for row in range(2):
            for column in range(2):
                quadrant_tokens=token_grid[:,row*half:(row+1)*half,column*half:(column+1)*half]
                quadrant_weights=weight_grid[:,row*half:(row+1)*half,column*half:(column+1)*half]
                quadrant_weights=quadrant_weights.flatten(1)
                quadrant_tokens=quadrant_tokens.flatten(1,2)
                quadrant_means.append((quadrant_tokens*quadrant_weights.unsqueeze(-1)).sum(dim=1)/quadrant_weights.sum(dim=1,keepdim=True).clamp_min(1e-6))
        return torch.cat(global_tokens+[patch_mean,patch_var]+quadrant_means,dim=1).float()
    def _predict_from_summary(self,summary):
        latent=self.feature_projection(summary.float())
        force_coeff=0.75*torch.tanh(self.force_head(latent).float())
        force=torch.matmul(force_coeff,self.force_basis.to(device=latent.device,dtype=latent.dtype).transpose(0,1))
        potential=torch.cat([torch.zeros_like(force[:,:1]),torch.cumsum(force,dim=1)],dim=1)
        temperature=0.28+0.42*torch.sigmoid(self.temperature_head(latent).float())
        direct_logits=0.15*torch.tanh(self.direct_residual_head(latent).float())
        direct_logits=direct_logits-direct_logits.mean(dim=1,keepdim=True)
        soft_prior=(self.prior_probs.to(device=summary.device,dtype=summary.dtype)+1e-3).pow(0.15)
        soft_prior=soft_prior/soft_prior.sum().clamp_min(1e-8)
        log_prior=soft_prior.clamp_min(1e-6).log().view(1,10)
        free_energy=self.free_energy_head(summary.float())
        logits=0.05*log_prior-free_energy-0.10*potential/temperature+direct_logits
        log_probs=F.log_softmax(logits/self.output_temperature,dim=1)
        if self.prior_mix>0.0:
            prior=self.prior_probs.to(device=summary.device,dtype=summary.dtype).view(1,10)
            probs=(1.0-self.prior_mix)*log_probs.exp()+self.prior_mix*prior
            log_probs=probs.clamp_min(1e-8).log()
        evidence=(1.5/temperature).clamp(0.05,20.0)
        alpha=self.evidence_base*self.prior_probs.to(device=summary.device,dtype=summary.dtype).view(1,10)+evidence*log_probs.exp()
        return log_probs,alpha
    def forward_from_features(self,features,return_pair=False):
        if features.ndim!=2 or features.shape[1]!=self.summary_dim:
            raise ValueError(f"features必须为(B,{self.summary_dim})")
        summary=(features.float()-self.feature_mean.to(device=features.device,dtype=torch.float32))/self.feature_std.to(device=features.device,dtype=torch.float32)
        if self.training:
            summary_scale=summary.detach().std(dim=0,keepdim=True).clamp_min(1e-3)
            summary=summary+0.025*torch.randn_like(summary)*summary_scale
            summary=F.dropout(summary,p=0.05,training=True)
        output_a=self._predict_from_summary(summary)
        if return_pair:
            output_b=self._predict_from_summary(summary)
            return (output_a[0],output_b[0]),(output_a[1],output_b[1])
        return output_a
    def forward(self,x,return_pair=False,return_tree=False,return_evidence=False):
        summary=x if x.ndim==2 else self.encode_features(x)
        outputs,concentrations=self.forward_from_features(summary,return_pair=return_pair)
        if return_evidence and return_tree:
            return outputs,{},concentrations
        if return_evidence:
            return outputs,concentrations
        if return_tree:
            return outputs,{}
        return outputs
    @torch.no_grad()
    def predict_proba(self,x):
        self.eval()
        output=self.forward(x)
        return output[0].exp() if isinstance(output,tuple) else output.exp()
class SparseLandauScoreFieldAblation(SparseLandauScoreField):
    def __init__(self,*args,use_force_field=True,use_free_energy=True,use_direct_residual=True,use_evidence=True,output_temperature=1.10,**kwargs):
        super().__init__(*args,**kwargs)
        self.use_force_field=bool(use_force_field)
        self.use_free_energy=bool(use_free_energy)
        self.use_direct_residual=bool(use_direct_residual)
        self.use_evidence=bool(use_evidence)
        self.output_temperature=float(output_temperature)
    def _predict_from_summary(self,summary):
        latent=self.feature_projection(summary.float())
        if self.use_force_field:
            force_coeff=0.75*torch.tanh(self.force_head(latent).float())
            force=torch.matmul(force_coeff,self.force_basis.to(device=latent.device,dtype=latent.dtype).transpose(0,1))
            potential=torch.cat([torch.zeros_like(force[:,:1]),torch.cumsum(force,dim=1)],dim=1)
        else:
            potential=torch.zeros((summary.shape[0],10),device=summary.device,dtype=summary.dtype)
        temperature=0.28+0.42*torch.sigmoid(self.temperature_head(latent).float())
        if self.use_direct_residual:
            direct_logits=0.15*torch.tanh(self.direct_residual_head(latent).float())
            direct_logits=direct_logits-direct_logits.mean(dim=1,keepdim=True)
        else:
            direct_logits=torch.zeros((summary.shape[0],10),device=summary.device,dtype=summary.dtype)
        soft_prior=(self.prior_probs.to(device=summary.device,dtype=summary.dtype)+1e-3).pow(0.15)
        soft_prior=soft_prior/soft_prior.sum().clamp_min(1e-8)
        log_prior=soft_prior.clamp_min(1e-6).log().view(1,10)
        if self.use_free_energy:
            free_energy=self.free_energy_head(summary.float())
        else:
            free_energy=torch.zeros((summary.shape[0],10),device=summary.device,dtype=summary.dtype)
        logits=0.05*log_prior-free_energy-0.10*potential/temperature+direct_logits
        log_probs=F.log_softmax(logits/self.output_temperature,dim=1)
        if self.prior_mix>0.0:
            prior_value=self.prior_probs.to(device=summary.device,dtype=summary.dtype).view(1,10)
            probs=(1.0-self.prior_mix)*log_probs.exp()+self.prior_mix*prior_value
            log_probs=probs.clamp_min(1e-8).log()
        if self.use_evidence:
            evidence=(1.5/temperature).clamp(0.05,20.0)
            alpha=self.evidence_base*self.prior_probs.to(device=summary.device,dtype=summary.dtype).view(1,10)+evidence*log_probs.exp()
        else:
            alpha=torch.ones_like(log_probs.exp())
        return log_probs,alpha
class SparseLandauScoreFieldWithoutForce(SparseLandauScoreFieldAblation):
    def __init__(self,*args,**kwargs):
        kwargs['use_force_field']=False
        super().__init__(*args,**kwargs)
class SparseLandauScoreFieldWithoutFreeEnergy(SparseLandauScoreFieldAblation):
    def __init__(self,*args,**kwargs):
        kwargs['use_free_energy']=False
        super().__init__(*args,**kwargs)
class SparseLandauScoreFieldWithoutResidual(SparseLandauScoreFieldAblation):
    def __init__(self,*args,**kwargs):
        kwargs['use_direct_residual']=False
        super().__init__(*args,**kwargs)
class SparseLandauScoreFieldWithoutEvidence(SparseLandauScoreFieldAblation):
    def __init__(self,*args,**kwargs):
        kwargs['use_evidence']=False
        super().__init__(*args,**kwargs)
class SparseLandauScoreFieldTemperatureOne(SparseLandauScoreFieldAblation):
    def __init__(self,*args,**kwargs):
        kwargs['output_temperature']=1.0
        super().__init__(*args,**kwargs)

