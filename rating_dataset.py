from pathlib import Path
import math
import random
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
def build_sparse_target(scores, prior=None):
    bins=torch.arange(1,11,dtype=torch.float32)
    score_tensor=torch.as_tensor(scores,dtype=torch.float32).view(-1,1)
    n=max(1,int(score_tensor.shape[0]))
    sigma=0.65+1.10/math.sqrt(float(n))
    kernel=torch.exp(-0.5*((bins.view(1,-1)-score_tensor)/sigma).square())
    kernel=kernel/kernel.sum(dim=1,keepdim=True).clamp_min(1e-8)
    q=kernel.mean(dim=0)
    center=(q*bins).sum()
    tau=1.0+0.8/math.sqrt(float(n))
    center_prior=torch.exp(-torch.abs(bins-center)/tau)
    if prior is not None:
        prior=torch.as_tensor(prior,dtype=torch.float32)
        if prior.shape!=(10,) or not torch.isfinite(prior).all() or (prior<0).any() or prior.sum()<=0:
            raise ValueError("target_prior必须是非负有限的10维概率向量")
        prior=(prior+1e-4).pow(0.35)
        prior=prior/prior.sum().clamp_min(1e-8)
        center_prior=center_prior*prior.sqrt()
    center_prior=center_prior/center_prior.sum().clamp_min(1e-8)
    beta=0.12/math.sqrt(float(n))
    q=(1.0-beta)*q+beta*center_prior
    return q/q.sum().clamp_min(1e-8)
class RatingDataset(Dataset):
    def __init__(self,data_dir="data",mask_dir=None,indices=None,augment=False,repeat=1,image_size=560,soft_target=False,target_prior=None):
        data_dir=Path(data_dir)
        self.mask_dir=Path(mask_dir) if mask_dir is not None else None
        self.augment=bool(augment)
        self.repeat=int(repeat) if self.augment else 1
        self.image_size=int(image_size)
        self.soft_target=bool(soft_target)
        self.target_prior=None if target_prior is None else torch.as_tensor(target_prior,dtype=torch.float32).clone()
        if self.image_size<224 or self.image_size%14!=0:
            raise ValueError("image_size必须是大于等于224的14的倍数")
        if self.repeat<1:
            raise ValueError("repeat必须大于等于1")
        self.image_paths=sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
        if indices is not None:
            self.image_paths=[self.image_paths[int(i)] for i in indices]
        if not self.image_paths:
            raise ValueError(f"没有找到图片：{data_dir.resolve()}")
        for image_path in self.image_paths:
            if self.mask_dir is not None and not (self.mask_dir/(image_path.stem+".png")).is_file():
                raise FileNotFoundError(f"缺少前景掩码：{image_path.stem}.png")
            if not image_path.with_suffix(".txt").is_file():
                raise FileNotFoundError(f"缺少评分文件：{image_path.with_suffix('.txt')}")
    def __len__(self):
        return len(self.image_paths)*self.repeat
    def _augment_pair(self,image,mask):
        if random.random()<0.5:
            image=TF.hflip(image)
            if mask is not None:
                mask=TF.hflip(mask)
        if random.random()<0.35:
            angle=random.uniform(-4,4)
            translate=[random.randint(-int(image.width*0.02),int(image.width*0.02)),random.randint(-int(image.height*0.02),int(image.height*0.02))]
            scale=random.uniform(0.98,1.02)
            image=TF.affine(image,angle=angle,translate=translate,scale=scale,shear=[0.0,0.0],interpolation=InterpolationMode.BILINEAR,fill=(127,127,127))
            if mask is not None:
                mask=TF.affine(mask,angle=angle,translate=translate,scale=scale,shear=[0.0,0.0],interpolation=InterpolationMode.BILINEAR,fill=0)
        if random.random()<0.25:
            image=TF.adjust_brightness(image,random.uniform(0.95,1.05))
        if random.random()<0.25:
            image=TF.adjust_contrast(image,random.uniform(0.95,1.05))
        return image,mask
    def _read_scores(self,text_path):
        text=text_path.read_text(encoding="utf-8-sig").strip()
        text=text.removeprefix("[").removesuffix("]")
        items=[item for item in re.split(r"[,，\s]+",text) if item]
        scores=[]
        for item in items:
            try:
                score=Decimal(item)
            except InvalidOperation:
                raise ValueError(f"{text_path}中存在无法解析的评分：{item!r}") from None
            if not score.is_finite():
                raise ValueError(f"{text_path}中存在非有限评分：{item!r}")
            score=max(Decimal("1"),min(score,Decimal("10")))
            scores.append(int(score.quantize(Decimal("1"),rounding=ROUND_HALF_UP)))
        if not scores:
            raise ValueError(f"{text_path}中没有有效评分")
        return scores
    def __getitem__(self,index):
        image_path=self.image_paths[index%len(self.image_paths)]
        text_path=image_path.with_suffix(".txt")
        with Image.open(image_path) as image_file:
            image=image_file.convert("RGB").resize((self.image_size,self.image_size),Image.Resampling.BILINEAR)
        mask=None
        if self.mask_dir is not None:
            with Image.open(self.mask_dir/f"{image_path.stem}.png") as mask_file:
                mask=mask_file.convert("L").resize((self.image_size,self.image_size),Image.Resampling.BILINEAR)
        if self.augment and random.random()<0.5:
            image,mask=self._augment_pair(image,mask)
        array=np.array(image,dtype=np.float32)/255.0
        x=torch.from_numpy(array).permute(2,0,1).contiguous()
        if mask is not None:
            mask_array=np.array(mask,dtype=np.float32)/255.0
            x=torch.cat([x,torch.from_numpy(mask_array).unsqueeze(0)],dim=0)
        scores=self._read_scores(text_path)
        if self.soft_target:
            y=build_sparse_target(scores,self.target_prior)
        else:
            y=torch.bincount(torch.tensor(scores,dtype=torch.long)-1,minlength=10).float()
            y=y/y.sum().clamp_min(1e-8)
        return x,y
