#!/usr/bin/env python
"""
Train a U-Net to segment brick kilns by technology type on the MuTBriK dataset.

Paper: Arora et al. (2026), "Brick kilns across India: A temporal analysis of
brick kiln distribution and technology types", International Journal of Applied
Earth Observation and Geoinformation, 153, 105496.

Two losses are available:
  --loss cmr_ce  Confusion-Matrix-Regularized Cross-Entropy (paper default).
                 Plain CE during the warm-up epochs, then CE + lambda * expected
                 confusion cost. The cost matrix is rebuilt from the training
                 confusion matrix after every epoch and smoothed with an EMA.
  --loss ce      Plain cross-entropy (baseline in the paper's ablation).

Run directly:
    python train.py --data_dir /path/to/MuTBriK
or as a Weights & Biases sweep:
    wandb sweep configs/sweep.yaml && wandb agent <sweep-id>

Without a wandb account, set WANDB_MODE=disabled (or offline).
"""
import argparse
import ast
import csv
import logging
import os
from datetime import datetime

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from tqdm import tqdm

IGNORE_INDEX = 255
ENCODERS = {
    "unet_b7": "efficientnet-b7",
    "unet_b5": "efficientnet-b5",
    "unet_b4": "efficientnet-b4",
    "unet_res101": "resnet101",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def int_list(value):
    """Parse '[1, 2, 3]' or '1,2,3' into a list of ints (works with wandb sweeps)."""
    if isinstance(value, list):
        return value
    value = value.strip()
    parsed = ast.literal_eval(value) if value.startswith("[") else [int(v) for v in value.split(",")]
    return [int(v) for v in parsed]


def get_config():
    p = argparse.ArgumentParser(description="Train brick kiln type segmentation on MuTBriK")
    # data
    p.add_argument("--data_dir", required=True, help="MuTBriK root with Train/, Validation/, Test/")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--bands", type=int_list, default=[1, 2, 3, 4, 5, 6, 7, 8],
                   help="1-based channels: B2, B3, B4, B8, B11, B12, NDVI, NDBI")
    p.add_argument("--classes", type=int_list, default=[0, 1, 2, 3, 4, 5],
                   help="Mask values to train on; other values are mapped to background")
    p.add_argument("--crop_size", type=int, default=256)
    # model and optimization
    p.add_argument("--arch", choices=list(ENCODERS), default="unet_b7")
    p.add_argument("--encoder_weights", default="none",
                   help="'none' = train from scratch (paper default) or 'imagenet'")
    p.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    p.add_argument("--initial_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_epochs", type=int, default=250)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    # loss
    p.add_argument("--loss", choices=["cmr_ce", "ce"], default="cmr_ce")
    p.add_argument("--beta_penalty", type=float, default=5.0, help="Cost matrix scale (beta)")
    p.add_argument("--alpha_ema", type=float, default=0.2, help="EMA smoothing of the cost matrix")
    p.add_argument("--cost_clip", type=float, default=3.0, help="Upper bound for cost entries")
    p.add_argument("--lam", type=float, default=1.0, help="Weight of the confusion cost term (lambda)")
    # logging
    p.add_argument("--project", default="brick-type-seg")
    p.add_argument("--save_predictions", type=int, default=1,
                   help="1 = write test predictions as GeoTIFFs")
    args, _ = p.parse_known_args()

    wandb.init(project=args.project, config=vars(args))
    cfg = argparse.Namespace(**dict(wandb.config))  # sweep values override the defaults
    cfg.bands, cfg.classes = int_list(cfg.bands), int_list(cfg.classes)
    return cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def normalize_image(image):
    """Per-channel min-max normalization of one patch."""
    image = np.nan_to_num(image, nan=0)
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    return (image - min_val) / (max_val - min_val + 1e-6)


class KilnDataset(Dataset):
    def __init__(self, image_paths, mask_paths, bands, classes, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.bands = bands
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def map_classes(self, mask):
        """Map mask values in `classes` to 0..N-1; everything else becomes background (0)."""
        mapped = np.zeros_like(mask, dtype=np.int64)
        for idx, cls in enumerate(self.classes):
            mapped[mask == cls] = idx
        return mapped

    def __getitem__(self, idx):
        try:
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read(self.bands).astype(np.float32)
                image_meta = src.meta
                image_meta["img_path"] = self.image_paths[idx]
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask_meta = src.meta
        except Exception as e:
            logger.warning(f"Skipping index {idx}: {e}")
            return None, None, None, None

        image = normalize_image(image)
        mask = self.map_classes(mask)
        if self.transform:
            out = self.transform(image=image.transpose(1, 2, 0), mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask, image_meta, mask_meta


def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, None, None
    images, masks, image_meta, mask_meta = zip(*batch)
    return torch.stack(images), torch.stack(masks), image_meta, mask_meta


def list_split(data_dir, split):
    """Sorted image and mask paths of one split, so that images and masks pair up."""
    def tifs(folder):
        folder = os.path.join(data_dir, split, folder)
        return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".tif", ".tiff")))

    images, masks = tifs("Image"), tifs("Mask")
    if len(images) != len(masks):
        raise ValueError(f"{split}: {len(images)} images but {len(masks)} masks")
    return images, masks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def iou_stats_batch(outputs, masks, C, ignore_index=IGNORE_INDEX):
    """Per-class intersection and union for one batch."""
    preds = outputs.argmax(1)
    valid = masks != ignore_index
    inter = torch.zeros(C, device=outputs.device)
    union = torch.zeros(C, device=outputs.device)
    for c in range(C):
        p = (preds == c) & valid
        t = (masks == c) & valid
        inter[c] += (p & t).sum()
        union[c] += (p | t).sum()
    return inter, union


def mean_iou_from_stats(inter, union, exclude_classes=None):
    ious = torch.where(union > 0, inter / union, torch.nan)
    keep = torch.ones_like(union, dtype=torch.bool)
    if exclude_classes:
        keep[exclude_classes] = False
    return torch.nanmean(ious[keep]).item(), ious.cpu().tolist()


def confusion_matrix(model, loader, C, device, ignore_index=IGNORE_INDEX):
    """Pixel confusion matrix (rows = true class, columns = predicted class)."""
    M = torch.zeros(C, C, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device).squeeze(1).long()
            pred = model(x).argmax(1)
            valid = y != ignore_index
            M += torch.bincount(C * y[valid] + pred[valid], minlength=C * C).view(C, C)
    return M


def save_confusion_plot(conf, path, title, normalize=False):
    data = conf.float() / conf.sum(1, keepdim=True).clamp(min=1) if normalize else conf
    labels = [f"C{c}" for c in range(conf.size(0))]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.cpu().numpy(), annot=True, fmt=".2f" if normalize else ".0f", square=True,
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"label": "Row-normalized" if normalize else "Pixel count"})
    ax.set(title=title, xlabel="Predicted class", ylabel="True class")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CMR-CE loss
# ---------------------------------------------------------------------------
def build_cost_matrix(conf, beta, clip, eps=1e-3):
    """Cost of predicting class j for true class i, from the row-normalized confusion matrix."""
    C = conf.size(0)
    row_prob = (conf + eps) / (conf.sum(1, keepdim=True) + eps * C)
    cost = beta * row_prob
    cost.fill_diagonal_(0.0)              # correct predictions are handled by CE
    cost = cost / (cost.mean() + 1e-6)    # keep the global scale near 1
    return cost.clamp_max(clip)


def cmr_ce_loss(logits, targets, cost_matrix, lam, ignore_index=IGNORE_INDEX):
    """Cross-entropy + lambda * expected confusion cost under the predicted distribution."""
    ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=ignore_index)
    probs = F.softmax(logits, dim=1)                                          # (N, C, H, W)
    row_costs = cost_matrix[targets.clamp(0, cost_matrix.size(0) - 1)]       # (N, H, W, C)
    exp_cost = (row_costs * probs.permute(0, 2, 3, 1)).sum(-1)                # (N, H, W)
    loss_map = (ce + lam * exp_cost)[targets != ignore_index]
    return loss_map.mean() if loss_map.numel() > 0 else logits.sum() * 0.0


def ce_loss(logits, targets):
    return F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg.classes)
    logger.info(f"Device: {device} | config: {vars(cfg)}")

    run_name = f"{cfg.loss}_{cfg.arch}_bands{''.join(map(str, cfg.bands))}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, f"{cfg.arch}_best_model.pth")
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_mIoU", "val_loss", "val_mIoU", "val_mIoU_no_bg"])

    # data
    train_tf = A.Compose([
        A.RandomCrop(height=cfg.crop_size, width=cfg.crop_size),
        A.OneOf([A.RandomRotate90(p=1.0), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=0.8),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([ToTensorV2()])

    def make_loader(split, transform, batch_size, shuffle, workers):
        images, masks = list_split(cfg.data_dir, split)
        ds = KilnDataset(images, masks, cfg.bands, cfg.classes, transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                          num_workers=workers, pin_memory=True)

    train_loader = make_loader("Train", train_tf, cfg.batch_size, True, cfg.num_workers)
    val_loader = make_loader("Validation", eval_tf, 8, False, cfg.num_workers)
    test_loader = make_loader("Test", eval_tf, 1, False, max(1, cfg.num_workers // 2))

    # model and optimizer
    weights = None if str(cfg.encoder_weights).lower() == "none" else cfg.encoder_weights
    model = Unet(ENCODERS[cfg.arch], encoder_weights=weights, in_channels=len(cfg.bands),
                 classes=num_classes, activation=None).to(device)
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.initial_lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.initial_lr)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / cfg.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs - cfg.warmup_epochs)

    cost_matrix = torch.ones(num_classes, num_classes, device=device)
    ema_cost = cost_matrix.clone()

    def criterion_for(epoch):
        if cfg.loss == "ce" or epoch < cfg.warmup_epochs:
            return ce_loss
        return lambda logits, targets: cmr_ce_loss(logits, targets, cost_matrix, cfg.lam)

    best_iou = 0.0
    for epoch in range(cfg.n_epochs):
        criterion = criterion_for(epoch)
        compute_train_iou = (epoch + 1) % 5 == 0

        # train
        model.train()
        running_loss = 0.0
        inter_sum = torch.zeros(num_classes, device=device)
        union_sum = torch.zeros(num_classes, device=device)
        for images, masks, *_ in tqdm(train_loader, desc=f"Train {epoch + 1}", leave=False):
            images, masks = images.to(device), masks.to(device).squeeze(1).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if compute_train_iou:
                i, u = iou_stats_batch(outputs, masks, num_classes)
                inter_sum += i
                union_sum += u
        train_loss = running_loss / len(train_loader)
        (warmup_scheduler if epoch < cfg.warmup_epochs else cosine_scheduler).step()
        lr = optimizer.param_groups[0]["lr"]
        train_miou, train_ious = (mean_iou_from_stats(inter_sum, union_sum) if compute_train_iou
                                  else (None, [None] * num_classes))

        # update the cost matrix from the training confusion matrix
        if cfg.loss == "cmr_ce" and epoch >= cfg.warmup_epochs:
            conf = confusion_matrix(model, train_loader, num_classes, device)
            np.save(os.path.join(out_dir, f"conf_train_epoch{epoch + 1:03d}.npy"), conf.cpu().numpy())
            new_cost = build_cost_matrix(conf, beta=cfg.beta_penalty, clip=cfg.cost_clip)
            ema_cost = cfg.alpha_ema * new_cost + (1 - cfg.alpha_ema) * ema_cost
            cost_matrix.copy_(ema_cost)
            wandb.log({"cost_matrix_max": cost_matrix.max().item(),
                       "cost_matrix_mean": cost_matrix.mean().item()}, step=epoch)

        # validate
        model.eval()
        val_loss = 0.0
        inter_val = torch.zeros(num_classes, device=device)
        union_val = torch.zeros(num_classes, device=device)
        with torch.no_grad():
            for images, masks, *_ in tqdm(val_loader, desc=f"Val {epoch + 1}", leave=False):
                images, masks = images.to(device), masks.to(device).squeeze(1).long()
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                i, u = iou_stats_batch(outputs, masks, num_classes)
                inter_val += i
                union_val += u
        val_loss /= len(val_loader)
        val_miou, val_ious = mean_iou_from_stats(inter_val, union_val)
        val_miou_no_bg, _ = mean_iou_from_stats(inter_val, union_val, exclude_classes=[0])

        log = {"epoch": epoch + 1, "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
               "val_mIoU_all": val_miou, "val_mIoU_no_bg": val_miou_no_bg,
               **{f"c{c}_val_iou": val_ious[c] for c in range(num_classes)}}
        if compute_train_iou:
            log["train_mIoU"] = train_miou
            log.update({f"c{c}_train_iou": train_ious[c] for c in range(num_classes)})
        wandb.log(log, step=epoch)
        logger.info(f"E{epoch + 1:03d} TL:{train_loss:.4f} VL:{val_loss:.4f} "
                    f"VIoU:{val_miou:.3f} VIoU(no bg):{val_miou_no_bg:.3f} "
                    + " ".join(f"c{c}:{v:.3f}" for c, v in enumerate(val_ious)))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_miou if train_miou is not None else "",
                                    val_loss, val_miou, val_miou_no_bg])

        # keep the model with the best kiln-class mIoU (background excluded)
        if val_miou_no_bg > best_iou:
            best_iou = val_miou_no_bg
            torch.save(model.state_dict(), best_model_path)

    test(cfg, model, best_model_path, test_loader, criterion_for(cfg.n_epochs), num_classes, device, out_dir)
    wandb.finish()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test(cfg, model, best_model_path, loader, criterion, num_classes, device, out_dir):
    logger.info("Testing the best model ...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Pixel-level precision / recall / F1 over kiln pixels: background targets are ignored.
    def metric(cls, average):
        return cls(num_classes=num_classes, average=average, ignore_index=IGNORE_INDEX).to(device)
    metrics = {f"{name}_{avg}": metric(cls, avg)
               for name, cls in [("precision", MulticlassPrecision), ("recall", MulticlassRecall),
                                 ("f1", MulticlassF1Score)]
               for avg in ["none", "macro", "weighted"]}

    pred_dir = os.path.join(out_dir, "test_predictions")
    if cfg.save_predictions:
        os.makedirs(pred_dir, exist_ok=True)

    test_loss = 0.0
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for images, masks, img_meta, mask_meta in tqdm(loader, desc="Test", leave=False):
            images, masks = images.to(device), masks.to(device).squeeze(1).long()
            outputs = model(images)
            test_loss += criterion(outputs, masks).item()
            i, u = iou_stats_batch(outputs, masks, num_classes)
            inter += i
            union += u

            preds = outputs.argmax(1)
            valid = masks != IGNORE_INDEX
            conf += torch.bincount(num_classes * masks[valid] + preds[valid],
                                   minlength=num_classes ** 2).view(num_classes, num_classes)
            kiln_targets = masks.clone()
            kiln_targets[kiln_targets == 0] = IGNORE_INDEX
            for m in metrics.values():
                m.update(preds, kiln_targets)

            if cfg.save_predictions:
                probs = F.softmax(outputs, dim=1)
                for b in range(images.shape[0]):
                    name = os.path.splitext(os.path.basename(img_meta[b]["img_path"]))[0]
                    pred = preds[b].cpu().numpy().astype(np.uint8)
                    mask = masks[b].cpu().numpy()
                    meta = mask_meta[b].copy()
                    meta.update(count=num_classes, dtype="float32")
                    with rasterio.open(os.path.join(pred_dir, f"{name}_soft.tif"), "w", **meta) as dst:
                        dst.write(probs[b].cpu().numpy().astype(np.float32))
                    meta.update(count=1, dtype="uint8")
                    with rasterio.open(os.path.join(pred_dir, f"{name}_predicted.tif"), "w", **meta) as dst:
                        dst.write(pred, 1)
                    error = ((mask != IGNORE_INDEX) & (mask != pred)).astype(np.uint8)
                    with rasterio.open(os.path.join(pred_dir, f"{name}_error.tif"), "w", **meta) as dst:
                        dst.write(error, 1)

    test_loss /= len(loader)
    test_miou, test_ious = mean_iou_from_stats(inter, union)
    results = {k: m.compute().cpu() for k, m in metrics.items()}

    summary = {"test_loss": test_loss, "test_mIoU": test_miou,
               **{k: float(v) for k, v in results.items() if not k.endswith("_none")}}
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "test_metrics_summary.csv"), index=False)
    pd.DataFrame({
        "class_id": range(num_classes),
        "mask_value": cfg.classes,
        "iou": test_ious,
        "precision": results["precision_none"].tolist(),
        "recall": results["recall_none"].tolist(),
        "f1": results["f1_none"].tolist(),
    }).to_csv(os.path.join(out_dir, "test_class_metrics.csv"), index=False)

    np.save(os.path.join(out_dir, "conf_test.npy"), conf.cpu().numpy())
    save_confusion_plot(conf, os.path.join(out_dir, "conf_test.png"), "Confusion matrix - test (counts)")
    save_confusion_plot(conf, os.path.join(out_dir, "conf_test_norm.png"),
                        "Confusion matrix - test (row-normalized)", normalize=True)

    wandb.log({**summary,
               **{f"c{c}_test_iou": test_ious[c] for c in range(num_classes)},
               **{f"c{c}_{name}": results[f"{name}_none"][c].item()
                  for name in ["precision", "recall", "f1"] for c in range(num_classes)}})
    logger.info(f"Test mIoU: {test_miou:.3f} | " + " ".join(f"c{c}:{v:.3f}" for c, v in enumerate(test_ious)))
    logger.info(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
