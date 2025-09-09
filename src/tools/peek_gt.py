#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
peek_gt.py — Visualize GT boxes and class names from your config's data loaders.
No model predictions. Uses the actual train/val/test dataloaders built from cfg.

Usage:
  python tools/peek_gt.py \
    configs/yolo_w/flir/yolo_world_s_modprompt_resnet34_flirv2_ir.py \
    --work-dir tmp_peek \
    --num-batches 2 \
    --stages train val test \
    --force-single-worker \
    --no-mosaic
"""

import os
import os.path as osp
import argparse
from typing import Optional, List

import numpy as np
import torch

from PIL import Image

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.visualization import Visualizer


def _to_disp_img(t: torch.Tensor) -> np.ndarray:
    """
    Convert preprocessed tensor (C,H,W) to uint8 HxWx3 for visualization.
    Handles grayscale and 0..1 vs 0..255 ranges heuristically.
    """
    if not torch.is_tensor(t):
        raise TypeError('expected torch.Tensor for image')

    x = t.detach().cpu().float()
    mn, mx = float(x.min()), float(x.max())
    # If it looks like 0..1, scale to 0..255; else clamp into 0..255
    if mx <= 1.5:
        x = x * 255.0
    x = x.clamp(0, 255)

    # Expand grayscale to 3 channels
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    # Truncate >3 channels
    if x.shape[0] > 3:
        x = x[:3]

    img = x.permute(1, 2, 0).numpy().astype(np.uint8)
    return img


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_image(arr: np.ndarray, out_path: str) -> None:
    Image.fromarray(arr).save(out_path)


def _draw_one_sample(
    vis: Visualizer,
    img: np.ndarray,
    bboxes: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    classes: Optional[List],
    texts_preview: Optional[str],
    out_path: str,
) -> None:
    """
    Draw bboxes + (optional) class names; annotate with texts preview; save.
    """
    vis.set_image(img)

    if bboxes is not None and len(bboxes) > 0:
        str_labels = None
        if labels is not None and classes is not None:
            str_labels = []
            # map class ids to names; guard bounds
            for lid in labels.tolist():
                if 0 <= lid < len(classes):
                    str_labels.append(str(classes[lid]))
                else:
                    str_labels.append(str(lid))
        # mmengine Visualizer API expects already set image
        vis.draw_bboxes(bboxes=bboxes, labels=str_labels)

    if texts_preview:
        vis.draw_texts(texts_preview, (5, 18))

    rendered = vis.get_image()
    _save_image(rendered, out_path)


def _peek_loader(
    tag: str,
    dataloader,
    out_dir: str,
    classes: Optional[List[str]],
    num_batches: int = 1,
) -> None:
    """
    tag: 'train' | 'val' | 'test'
    dataloader: built from cfg
    out_dir: base output dir
    """
    save_dir = osp.join(out_dir, tag)
    _ensure_dir(save_dir)
    vis = Visualizer(name=f'{tag}_peek', save_dir=save_dir)

    it = iter(dataloader)
    for bi in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        imgs = batch['inputs']                           # (B,C,H,W)
        ds = batch['data_samples']                       # list[DetDataSample] OR dict (yolow_collate)
        B = imgs.shape[0]

        for i in range(B):
            img_np = _to_disp_img(imgs[i])

            bboxes_np = None
            labels_np = None
            texts_preview = None

            # Case A: default collate -> list of DetDataSample
            if isinstance(ds, (list, tuple)):
                s = ds[i]
                inst = getattr(s, 'gt_instances', None)
                if inst is not None and hasattr(inst, 'bboxes') and inst.bboxes is not None:
                    b = inst.bboxes
                    # Some versions wrap bboxes; get underlying tensor if present
                    if hasattr(b, 'tensor'):
                        b = b.tensor
                    bboxes_np = b.detach().cpu().numpy()
                if inst is not None and hasattr(inst, 'labels') and inst.labels is not None:
                    labels_np = inst.labels.detach().cpu().numpy()

                meta = getattr(s, 'metainfo', {})
                tx = meta.get('texts', None)
                if tx is not None:
                    # flatten [[...]] to [...]
                    if len(tx) > 0 and isinstance(tx[0], list):
                        flat = sum(tx, [])
                    else:
                        flat = tx
                    texts_preview = "texts[{}]: {}".format(len(flat), flat[:5])

            # Case B: yolow_collate -> dict with 'bboxes_labels' and 'texts'
            elif isinstance(ds, dict):
                bl = ds.get('bboxes_labels', None)
                if bl is not None and len(bl) > i and bl[i] is not None:
                    arr = bl[i]
                    # Torch tensor or numpy
                    if torch.is_tensor(arr):
                        arr = arr.detach().cpu().numpy()
                    else:
                        arr = np.asarray(arr)
                    if arr.size > 0:
                        if arr.shape[1] >= 4:
                            bboxes_np = arr[:, :4]
                            labels_np = arr[:, 4].astype(np.int32) if arr.shape[1] >= 5 else None

                txs = ds.get('texts', None)
                if txs is not None and len(txs) > i:
                    t = txs[i]
                    if len(t) > 0 and isinstance(t[0], list):
                        flat = sum(t, [])
                    else:
                        flat = t
                    texts_preview = "texts[{}]: {}".format(len(flat), flat[:5])

            out_path = osp.join(save_dir, "batch{:02d}_idx{:02d}.jpg".format(bi, i))
            _draw_one_sample(vis, img_np, bboxes_np, labels_np, classes, texts_preview, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='Path to config .py')
    ap.add_argument('--work-dir', default='work_dirs/peek_gt', help='Base output dir')
    ap.add_argument('--num-batches', type=int, default=1, help='Batches per stage to visualize')
    ap.add_argument('--stages', nargs='+', default=['train', 'val', 'test'],
                    choices=['train', 'val', 'test'])
    ap.add_argument('--force-single-worker', action='store_true',
                    help='Override num_workers=0 for all dataloaders (recommended for debug)')
    ap.add_argument('--no-mosaic', action='store_true',
                    help='Best-effort: switch to stage-2 pipeline early if available')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)

    # ensure work_dir exists for Runner
    if not cfg.get('work_dir', None):
        cfg.work_dir = args.work_dir or os.path.join(
            './work_dirs',
            os.path.splitext(os.path.basename(args.config))[0]
        )


    # Optional: reduce workers for debug stability
    if args.force_single_worker:
        if 'train_dataloader' in cfg:
            cfg.train_dataloader.setdefault('num_workers', 0)
            cfg.train_dataloader['num_workers'] = 0
            cfg.train_dataloader.setdefault('persistent_workers', False)
            cfg.train_dataloader['persistent_workers'] = False
        if 'val_dataloader' in cfg:
            cfg.val_dataloader.setdefault('num_workers', 0)
            cfg.val_dataloader['num_workers'] = 0
        if 'test_dataloader' in cfg:
            cfg.test_dataloader.setdefault('num_workers', 0)
            cfg.test_dataloader['num_workers'] = 0

    # Optional: disable mosaic by forcing switch hook at epoch 0 (if present)
    if args.no_mosaic and 'custom_hooks' in cfg and isinstance(cfg.custom_hooks, list):
        for h in cfg.custom_hooks:
            if isinstance(h, dict) and h.get('type', '').endswith('PipelineSwitchHook'):
                h['switch_epoch'] = 0

    # Build runner to reuse EXACT dataloaders from the config
    runner = Runner.from_cfg(cfg)

    # Pull classes (if provided)
    classes = None
    try:
        ds0 = runner.train_dataloader.dataset
        inner = getattr(ds0, 'dataset', ds0)  # unwrap MultiModalDataset
        meta = getattr(inner, 'metainfo', None)
        if meta and 'classes' in meta and meta['classes'] is not None:
            classes = list(meta['classes'])
    except Exception as e:
        print_log('Could not read classes from dataset: {}'.format(e), logger='current')

    out_dir = osp.join(cfg.get('work_dir', args.work_dir), 'peek_gt')
    os.makedirs(out_dir, exist_ok=True)

    if 'train' in args.stages:
        _peek_loader('train', runner.train_dataloader, out_dir, classes, args.num_batches)
        print_log('Train samples saved to {}'.format(osp.join(out_dir, 'train')), logger='current')

    if 'val' in args.stages:
        _peek_loader('val', runner.val_dataloader, out_dir, classes, args.num_batches)
        print_log('Val samples saved to {}'.format(osp.join(out_dir, 'val')), logger='current')

    if 'test' in args.stages:
        _peek_loader('test', runner.test_dataloader, out_dir, classes, args.num_batches)
        print_log('Test samples saved to {}'.format(osp.join(out_dir, 'test')), logger='current')


if __name__ == '__main__':
    main()
