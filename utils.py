import torch
from PIL import Image
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib.pyplot as plt
def predict_one_sample(img, model, y_gt=None, device=torch.device("cpu"), mask=None):
    img = img.convert("RGB")
    img = img.resize((560, 560), Image.Resampling.BILINEAR)
    array = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)
    if mask is not None:
        mask_array = np.array(mask.convert("L").resize((560, 560), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        x = torch.cat([x, torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)], dim=1)
    model = model.to(device)
    model.eval()
    x = x.to(device)
    if y_gt is not None:
        y_gt = y_gt.to(device)
        print(f"真实标签为: {y_gt}")
    with torch.no_grad():
        logits = model(x)
        y_pred = torch.softmax(logits, dim=1).squeeze(0)
    print(f"预测概率分布为: {y_pred}")
    print(f"概率之和为: {y_pred.sum().item():.6f}")
    scores = torch.arange(1, 11, device=device, dtype=y_pred.dtype)
    pred_mean = (y_pred * scores).sum().item()
    print(f"预测平均评分为: {pred_mean:.4f}")
    return y_pred
def read_one_y_gt(y_path):
    # txt 内容例如：5,6,6,2 或 [4.5,6,6,2]
    text = Path(y_path).read_text(encoding="utf-8-sig").strip()
    text = text.removeprefix("[").removesuffix("]")
    scores = []
    for item in re.split(r"[,，\s]+", text):
        if item:
            score = Decimal(item)
            if not score.is_finite() or not (1 <= score <= 10):
                raise ValueError(f"评分必须在 1~10 之间：{item}")
            scores.append(
                int(score.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            )
    
    if not scores:
        raise ValueError("txt 中没有评分")
    y = torch.bincount(
        torch.tensor(scores, dtype=torch.long) - 1,
        minlength=10,
    ).float()
    y = y / y.sum()
    return y
def draw_train_val_loss(
    lst_train_emd,
    lst_train_mae,
    lst_val_emd,
    lst_val_mae,
    lst_train_loss,
    lst_val_loss,
    lst_train_nll,
    lst_val_nll,
    lst_train_spearman,
    lst_val_spearman,
    save=False
):
    metrics = [
        ("Loss", lst_train_loss, lst_val_loss),
        ("EMD", lst_train_emd, lst_val_emd),
        ("MAE", lst_train_mae, lst_val_mae),
        ("NLL", lst_train_nll, lst_val_nll),
        ("Spearman", lst_train_spearman, lst_val_spearman),
    ]

    # 检查列表长度
    num_epochs = len(lst_train_emd)
    if num_epochs == 0:
        raise ValueError("没有训练数据")

    for name, train, val in metrics:
        if len(train) != num_epochs or len(val) != num_epochs:
            raise ValueError(f"{name} 的数据长度不一致")

    epochs = range(1, num_epochs + 1)

    # 创建 2 行 3 列子图
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, (name, train, val) in enumerate(metrics):
        ax = axes[i]

        # 训练集：实线
        ax.plot(
            epochs, train,
            color="tab:blue",
            linestyle="-",
            linewidth=2,
            label=f"Train {name}"
        )

        # 验证集：虚线
        ax.plot(
            epochs, val,
            color="tab:orange",
            linestyle="--",
            linewidth=2,
            label=f"Val {name}"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(f"Training and Validation {name}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        # Epoch 显示为整数
        ax.xaxis.get_major_locator().set_params(integer=True)

    # 隐藏第六个空白子图
    axes[5].axis("off")

    fig.suptitle(
        "Training and Validation Metrics",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout()

    if save:
        save_path = Path("outputs/training_metrics.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=600,
            bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)

def preprocess_face_dataset(src_dir="data/train_data", dst_dir="data/face_data", mask_dir="data/face_masks", size=560, expand=1.7):
    import cv2
    import shutil
    from PIL import ImageOps, ImageFilter
    from facexlib.detection import init_detection_model
    from facexlib.parsing import init_parsing_model
    src_dir, dst_dir, mask_dir = Path(src_dir), Path(dst_dir), Path(mask_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    process_device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = init_detection_model("retinaface_resnet50", device=process_device)
    parser = init_parsing_model("bisenet", device=process_device)
    detector.eval()
    parser.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=process_device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=process_device).view(1, 3, 1, 1)
    for src in sorted(src_dir.iterdir()):
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        target, target_mask, target_txt = dst_dir / src.name, mask_dir / f"{src.stem}.png", dst_dir / f"{src.stem}.txt"
        if target.is_file() and target_mask.is_file() and target_txt.is_file():
            continue
        with Image.open(src) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        scale = min(1.0, 1024.0 / max(image.width, image.height))
        view = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
        with torch.no_grad():
            boxes = detector.detect_faces(view, 0.7)
        if len(boxes) == 1:
            x1, y1, x2, y2 = [float(v) / scale for v in boxes[0][:4]]
            side = max(1, int(round(max(x2 - x1, y2 - y1) * expand)))
            left = int(round((x1 + x2 - side) / 2))
            top = int(round((y1 + y2 - side) / 2 - 0.1 * (y2 - y1)))
            crop = Image.new("RGB", (side, side), (127, 127, 127))
            crop.paste(image, (-left, -top))
            crop = crop.resize((size, size), Image.Resampling.LANCZOS)
            parsing_image = crop.resize((512, 512), Image.Resampling.BILINEAR)
            inp = torch.from_numpy(np.asarray(parsing_image, dtype=np.float32).transpose(2, 0, 1).copy()).unsqueeze(0).to(process_device) / 255.0
            inp = (inp - mean) / std
            with torch.no_grad():
                classes = parser(inp)[0].argmax(dim=1)[0].cpu().numpy()
            mask_np = np.isin(classes, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]).astype(np.uint8) * 255
            mask_np = cv2.GaussianBlur(mask_np, (0, 0), 2.5)
            mask = Image.fromarray(mask_np).resize((size, size), Image.Resampling.BILINEAR)
            if np.asarray(mask).mean() < 10:
                mask = Image.new("L", (size, size), 255)
            result = Image.composite(crop, crop.filter(ImageFilter.GaussianBlur(radius=12)), mask)
        else:
            result = ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS, color=(127, 127, 127))
            mask = Image.new("L", (size, size), 255)
            print(f"需要人工检查：{src.name}，检测到 {len(boxes)} 张脸，暂时保留整图")
        result.save(target, quality=95)
        mask.save(target_mask)
        shutil.copy2(src.with_suffix(".txt"), target_txt)
    del detector, parser
    if process_device == "cuda":
        torch.cuda.empty_cache()
    print(f"预处理完成：{dst_dir}")
def visualize_sample(img_id,model,device):
    img = Image.open(f"data/face_data/{img_id}.jpg")
    mask = Image.open(f"data/face_masks/{img_id}.png")
    y_gt = read_one_y_gt(f"data/face_data/{img_id}.txt")
    res = predict_one_sample(img, model, y_gt=y_gt, device=device, mask=mask)
    scores = np.arange(1, 11)
    pred = res.detach().cpu().numpy().flatten()
    gt = y_gt.detach().cpu().numpy().flatten()
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(scores - width / 2, gt, width=width, color="tab:orange", label="Ground Truth")
    ax.bar(scores + width / 2, pred, width=width, color="tab:blue", label="Prediction")
    ax.set_xlabel("Rating Score")
    ax.set_ylabel("Probability")
    ax.set_title("Predicted vs Ground Truth Rating Distribution")
    ax.set_xticks(scores)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()