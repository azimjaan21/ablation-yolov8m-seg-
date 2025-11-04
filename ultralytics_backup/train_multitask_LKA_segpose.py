#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint Multitask YOLOv8 – LKALite Shared Backbone
=================================================
Trains segmentation (glove / no-glove) and wrist-pose keypoints
in one forward pass with shared LKALite backbone.

Loss = L_seg + λ * L_pose
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
import sys, os

from ultralytics.data.build import build_dataloader, build_yolo_dataset
import yaml
from types import SimpleNamespace

from pathlib import Path
import os, time

# =====================================================
# === CONFIG ==========================================
# =====================================================
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_SIZE      = 640
EPOCHS        = 100
BATCH_SIZE    = 8
LAMBDA_POSE   = 0.4  # relative weight of pose loss
LR            = 1e-4
PROJECT_ROOT  = Path("exp_multitask_joint")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

# Default paths to datasets and models (can be overridden via CLI)
SEG_YAML_DEFAULT  = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\gloves.yaml"
POSE_YAML_DEFAULT = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml"
YAML_LKALITE_DEFAULT = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\models\v8\LKALite_p3.yaml"
POSE_PRETRAIN_DEFAULT = r"yolov8m-pose.pt"

# =====================================================
# === JOINT MODEL =====================================
# =====================================================
class MultitaskLKALite(nn.Module):
    """Shared LKALite backbone + 2 heads (seg + pose)"""
    def __init__(self, seg_model: YOLO, pose_model: YOLO, bb_end=6):
        super().__init__()
        seg_raw, pose_raw = seg_model.model, pose_model.model
        # Use the segmentation model's backbone modules as the shared backbone. Both models were
        # constructed from the same LKALite yaml earlier so channel widths align.
        self.backbone = nn.ModuleList(list(seg_raw.model[:bb_end]))
        self.seg_head = nn.ModuleList(list(seg_raw.model[bb_end:]))
        self.pose_head = nn.ModuleList(list(pose_raw.model[bb_end:]))
        # Lazy adapters to align channel counts when backbone and head widths differ.
        # Keys are module indices in pose_head (stringified ints).
        self.pose_adapters = nn.ModuleDict()

    def forward(self, x):
        feats = []
        for m in self.backbone:
            fi = getattr(m, "f", -1)
            x_in = x if fi == -1 else (feats[fi] if isinstance(fi, int)
                    else [feats[j] if j != -1 else x for j in fi])
            x = m(x_in) if isinstance(x_in, list) else m(x_in)
            feats.append(x)

        # --- Two heads ---
        seg_out, pose_out = x, x
        seg_feats = feats.copy()
        pose_feats = feats.copy()

        for m in self.seg_head:
            fi = getattr(m, "f", -1)
            x_in = seg_out if fi == -1 else (seg_feats[fi] if isinstance(fi, int)
                    else [seg_feats[j] if j != -1 else seg_out for j in fi])
            seg_out = m(x_in) if isinstance(x_in, list) else m(x_in)
            seg_feats.append(seg_out)

        # Iterate with index so we can attach adapters for specific modules when needed
        for i, m in enumerate(self.pose_head):
            fi = getattr(m, "f", -1)
            x_in = pose_out if fi == -1 else (pose_feats[fi] if isinstance(fi, int)
                    else [pose_feats[j] if j != -1 else pose_out for j in fi])

            # Helper to find expected in_channels of module `m` by inspecting its first 4D weight
            def _expected_in_channels(mod):
                for p in mod.parameters():
                    if p.dim() == 4:
                        return p.shape[1]
                return None

            # If input is a list (multiple feature maps), just pass it through as usual
            if isinstance(x_in, list):
                try:
                    pose_out = m(x_in)
                except RuntimeError:
                    # try adapting only the first tensor if necessary
                    first = x_in[0]
                    exp = _expected_in_channels(m)
                    if exp is not None and first is not None:
                        cur_c = first.shape[1]
                        if cur_c != exp:
                            key = str(i)
                            if key not in self.pose_adapters:
                                self.pose_adapters[key] = nn.Conv2d(cur_c, exp, kernel_size=1)
                                self.pose_adapters[key].to(first.device)
                                # match dtype
                                try:
                                    self.pose_adapters[key].to(dtype=first.dtype)
                                except Exception:
                                    pass
                            # replace first tensor with adapted
                            x_in = [self.pose_adapters[key](first)] + x_in[1:]
                            pose_out = m(x_in)
                        else:
                            raise
                    else:
                        raise
            else:
                # x_in is a tensor
                try:
                    pose_out = m(x_in)
                except RuntimeError as e:
                    # Infer expected in-channels and create a 1x1 adapter if mismatch occurs
                    exp = _expected_in_channels(m)
                    if exp is None:
                        raise
                    cur_c = x_in.shape[1]
                    if cur_c != exp:
                        key = str(i)
                        if key not in self.pose_adapters:
                            self.pose_adapters[key] = nn.Conv2d(cur_c, exp, kernel_size=1)
                            self.pose_adapters[key].to(x_in.device)
                            try:
                                self.pose_adapters[key].to(dtype=x_in.dtype)
                            except Exception:
                                pass
                        x_in = self.pose_adapters[key](x_in)
                        pose_out = m(x_in)
                    else:
                        raise

            pose_feats.append(pose_out)

        return seg_out, pose_out


# =====================================================
# === TRAINING LOOP ===================================
# =====================================================
def train_joint(seg_yaml: str, pose_yaml: str, yaml_lkalite: str, pose_pretrain: str,
                epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                img_size: int = IMG_SIZE, lambda_pose: float = LAMBDA_POSE,
                lr: float = LR, device: str = DEVICE, dry_run: bool = False):
    """Train with a shared LKALite backbone. Paths/params come from CLI or defaults."""
    print("🚀 Loading models...")
    if not Path(yaml_lkalite).exists():
        raise FileNotFoundError(f"LKALite YAML not found: {yaml_lkalite}")
    # Build segmentation model from LKALite yaml
    seg_yolo = YOLO(yaml_lkalite)

    # For the pose model prefer loading from the provided pretrained pose weights (if they exist)
    # so the model.task and loss type are correct for pose. Otherwise fall back to LKALite yaml.
    if Path(pose_pretrain).exists():
        try:
            pose_yolo = YOLO(pose_pretrain)
            print(f"✅ Loaded pose model from pretrained weights: {pose_pretrain}")
        except Exception as e:
            print(f"⚠️ Could not initialize YOLO from pose_pretrain ({pose_pretrain}): {e}; falling back to LKALite yaml")
            pose_yolo = YOLO(yaml_lkalite)
    else:
        pose_yolo = YOLO(yaml_lkalite)

    # Force the pose model to use the 'pose' task so the correct loss (keypoints) is selected.
    try:
        if getattr(pose_yolo.model, 'task', None) != 'pose':
            pose_yolo.model.task = 'pose'
        # Also ensure model.args.task if present
        if hasattr(pose_yolo.model, 'args'):
            try:
                setattr(pose_yolo.model.args, 'task', 'pose')
            except Exception:
                pass
    except Exception:
        pass
    print("✅ Model built with shared LKALite backbone")

    # ✅ Build dataloaders using ultralytics.yolo.data.utils
    print("📦 Building dataloaders...")
    # Load dataset YAMLs
    with open(seg_yaml, 'r') as f:
        seg_data = yaml.safe_load(f)
    with open(pose_yaml, 'r') as f:
        pose_data = yaml.safe_load(f)

    # Ensure required keys in dataset dicts
    seg_data.setdefault('channels', 3)
    pose_data.setdefault('channels', 3)

    # Minimal cfg used by build_yolo_dataset. Provide common hyp keys expected by transforms.
    default_hyp = dict(
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
    # mask_ratio must be > 0 to avoid division by zero in mask utils; 1 = no downsample
    # Use integer 1 (not float) so downstream integer divisions produce integer sizes for cv2.resize
    mask_ratio=1,
    overlap_mask=0.0,
        bgr=0.0,
        # affine / mosaic params used by v8_transforms
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        # HSV / color
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        # flips
        fliplr=0.5,
        flipud=0.0,
    )

    # Some Ultralytics versions set model.args as a dict; loss implementations expect a Namespace
    # with attributes like `overlap_mask`, `mask_ratio`, etc. Normalize these to a SimpleNamespace.
    def _ensure_model_args(yolo_obj, defaults: dict):
        try:
            mm = yolo_obj.model
        except Exception:
            return
        args = getattr(mm, "args", None)
        if isinstance(args, dict):
            merged = {**defaults, **args}
            mm.args = SimpleNamespace(**merged)
        elif args is None:
            mm.args = SimpleNamespace(**defaults)
        else:
            # args exists as Namespace-like; ensure defaults present
            for k, v in defaults.items():
                if not hasattr(mm.args, k):
                    setattr(mm.args, k, v)

    # Defaults used by transforms/loss
    _ensure_model_args(seg_yolo, default_hyp)
    _ensure_model_args(pose_yolo, default_hyp)

    # Ensure the YOLO model modules (and any loss buffers created on first use) are on the target device.
    # This avoids device mismatches where preds are on CUDA but loss-internal tensors (e.g. proj) are on CPU.
    try:
        seg_yolo.model.to(device)
    except Exception:
        pass
    try:
        pose_yolo.model.to(device)
    except Exception:
        pass

    model = MultitaskLKALite(seg_yolo, pose_yolo).to(device)
    # ModuleDict to hold adapters that remap pose prediction channels to dataset keypoint channels
    model.kpt_adapters = nn.ModuleDict()
    model.kpt_adapters.to = lambda *a, **k: None  # safe-nop if attempted; actual adapters moved when created

    seg_cfg = SimpleNamespace(
        imgsz=img_size,
        rect=False,
        cache=False,
        single_cls=False,
        task=seg_data.get('task', 'segment'),
        classes=None,
        fraction=1.0,
        **default_hyp,
    )

    pose_cfg = SimpleNamespace(
        imgsz=img_size,
        rect=False,
        cache=False,
        single_cls=False,
        task=pose_data.get('task', 'pose'),
        classes=None,
        fraction=1.0,
        **default_hyp,
    )

    # Build YOLO datasets then dataloaders
    seg_dataset = build_yolo_dataset(seg_cfg, seg_data.get('train', seg_yaml), batch_size, seg_data, mode='train')
    pose_dataset = build_yolo_dataset(pose_cfg, pose_data.get('train', pose_yaml), batch_size, pose_data, mode='train')
    workers = min(4, os.cpu_count() or 1)
    seg_loader = build_dataloader(seg_dataset, batch_size, workers, shuffle=True)
    pose_loader = build_dataloader(pose_dataset, batch_size, workers, shuffle=True)
    steps = min(len(seg_loader), len(pose_loader))
    if dry_run:
        # run just one step per epoch for quick validation
        epochs = 1
        steps = 1

    opt = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    print(f"🎯 Training for {epochs} epochs on {steps} batches per epoch")

    for epoch in range(epochs):
        model.train()
        seg_iter, pose_iter = iter(seg_loader), iter(pose_loader)
        t0 = time.time()

        for step in range(steps):
            # Dataloader yields a batch dict (see YOLODataset.collate_fn).
            batch_seg = next(seg_iter)
            batch_pose = next(pose_iter)

            # Move tensor entries in the batch to device
            def move_batch(batch):
                for k, v in list(batch.items()):
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                return batch

            batch_seg = move_batch(batch_seg)
            batch_pose = move_batch(batch_pose)

            # Quick dataset sanity checks and helpful errors
            # If segmentation model expects masks but batch doesn't provide them, raise a clear error
            seg_task = getattr(seg_cfg, 'task', 'segment') if 'seg_cfg' in locals() else 'segment'
            pose_task = getattr(pose_cfg, 'task', 'pose') if 'pose_cfg' in locals() else 'pose'
            if seg_task == 'segment' and 'masks' not in batch_seg:
                raise TypeError(
                    "Segmentation batch missing 'masks' key.\n"
                    "This usually means the dataset yaml for segmentation is not a 'segment' dataset or labels/masks are missing.\n"
                    "Check your segmentation dataset yaml and that masks are generated in the labels folder.\n"
                    f"seg_yaml={seg_yaml} | batch keys={list(batch_seg.keys())}"
                )
            if pose_task == 'pose' and 'keypoints' not in batch_pose:
                raise TypeError(
                    "Pose batch missing 'keypoints' key.\n"
                    "This usually means the dataset yaml for pose is not a 'pose' dataset or keypoint labels are missing.\n"
                    "Check your pose dataset yaml and that keypoint labels are present in the labels folder.\n"
                    f"pose_yaml={pose_yaml} | batch keys={list(batch_pose.keys())}"
                )

            imgs_seg = batch_seg["img"]
            imgs_pose = batch_pose["img"]

            # Ensure input images are floating and match model parameter dtype (float32 or float16).
            # Some dataloaders may return uint8 ByteTensors; convert and normalize to [0,1].
            def prep_img(img_tensor, model_ref):
                # Move to float if needed
                if not torch.is_floating_point(img_tensor):
                    img_tensor = img_tensor.float()
                # Normalize if still in 0-255 range
                try:
                    if img_tensor.max() > 1.0:
                        img_tensor = img_tensor / 255.0
                except Exception:
                    # if tensor empty or non-numeric, skip
                    pass
                # Cast to model param dtype
                target_dtype = next(model_ref.parameters()).dtype
                if img_tensor.dtype != target_dtype:
                    img_tensor = img_tensor.to(dtype=target_dtype)
                return img_tensor

            imgs_seg = prep_img(imgs_seg, model)
            imgs_pose = prep_img(imgs_pose, model)

            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                seg_pred, _ = model(imgs_seg)
                # YOLO.loss expects (batch_dict, preds=None) and returns (loss, loss_items)
                loss_seg_total, _ = seg_yolo.loss(batch_seg, preds=seg_pred)

                _, pose_pred = model(imgs_pose)
                # Adapt pose_pred channels to match dataset keypoint shape if necessary.
                adapted_pose_pred = pose_pred
                try:
                    # helper to extract feats and pred_kpts from various possible wrappers
                    def _extract(pred):
                        # Case A: pred is (feats, pred_kpts)
                        if isinstance(pred, (list, tuple)) and len(pred) == 2 and isinstance(pred[0], list):
                            return 'outer', pred[0], pred[1]
                        # Case B: pred is (something, (feats, pred_kpts))
                        if isinstance(pred, (list, tuple)) and len(pred) > 1:
                            inner = pred[1]
                            if isinstance(inner, (list, tuple)) and len(inner) == 2:
                                return 'inner', inner[0], inner[1]
                        # Case C: pred itself is (feats, pred_kpts)
                        if isinstance(pred, (list, tuple)) and len(pred) == 2:
                            return 'outer', pred[0], pred[1]
                        return None, None, None

                    loc, feats_p, pred_kpts = _extract(pose_pred)
                    if pred_kpts is not None and isinstance(pred_kpts, torch.Tensor):
                        C = pred_kpts.shape[1]
                        kpt_shape = pose_data.get('kpt_shape', None)
                        if kpt_shape is not None:
                            desired_C = int(kpt_shape[0]) * int(kpt_shape[1])
                            if C != desired_C:
                                # create or reuse adapter
                                key = 'kpt'
                                if key not in model.kpt_adapters:
                                    adapter = nn.Conv1d(C, desired_C, kernel_size=1).to(device)
                                    model.kpt_adapters[key] = adapter
                                    # If optimizer already exists, add adapter parameters so they will be trained
                                    try:
                                        opt.add_param_group({'params': adapter.parameters()})
                                    except Exception:
                                        pass
                                else:
                                    adapter = model.kpt_adapters[key]
                                # ensure adapter is on correct device/dtype
                                adapter.to(device)
                                try:
                                    adapter.to(dtype=pred_kpts.dtype)
                                except Exception:
                                    pass
                                # pred_kpts shape expected (batch, C, grids)
                                pred_kpts = adapter(pred_kpts)
                                # reconstruct adapted pose_pred in same structure
                                if loc == 'outer':
                                    adapted_pose_pred = (feats_p, pred_kpts) if isinstance(pose_pred, tuple) else [feats_p, pred_kpts]
                                elif loc == 'inner':
                                    new = list(pose_pred)
                                    new[1] = (feats_p, pred_kpts)
                                    adapted_pose_pred = tuple(new) if isinstance(pose_pred, tuple) else new
                except Exception:
                    adapted_pose_pred = pose_pred

                loss_pose_total, _ = pose_yolo.loss(batch_pose, preds=adapted_pose_pred)

                loss = loss_seg_total + lambda_pose * loss_pose_total

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % 10 == 0:
                print(
                    f"[Epoch {epoch+1}/{epochs}] Step {step}/{steps} | "
                    f"L_seg={loss_seg_total.item():.4f} L_pose={loss_pose_total.item():.4f} Total={loss.item():.4f}"
                )

        dt = time.time() - t0
        torch.save(model.state_dict(), PROJECT_ROOT / f"epoch_{epoch+1:03d}.pt")
        print(f"✅ Epoch {epoch+1}/{epochs} completed in {dt:.1f}s\n")

    print("🎉 Multitask joint training finished.")


# =====================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train LKALite shared backbone for seg+pose")
    p.add_argument("--seg-yaml", default=SEG_YAML_DEFAULT, help="segmentation dataset yaml path")
    p.add_argument("--pose-yaml", default=POSE_YAML_DEFAULT, help="pose dataset yaml path")
    p.add_argument("--lka-yaml", default=YAML_LKALITE_DEFAULT, help="LKALite model yaml")
    p.add_argument("--pose-weights", default=POSE_PRETRAIN_DEFAULT, help="pose pretrained weights or yaml")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--lambda-pose", type=float, default=LAMBDA_POSE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--dry-run", action="store_true", help="Run a single-batch dry run to validate training loop")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_joint(seg_yaml=args.seg_yaml,
                pose_yaml=args.pose_yaml,
                yaml_lkalite=args.lka_yaml,
                pose_pretrain=args.pose_weights,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                lambda_pose=args.lambda_pose,
                lr=args.lr,
                device=args.device,
                dry_run=args.dry_run)
