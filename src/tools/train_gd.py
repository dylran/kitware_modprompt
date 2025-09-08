# RemovePoliceCVPR
import argparse
import glob
import os
import os.path as osp
import re

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.logging import print_log

from mmdet.utils import setup_cache_size_limit_of_dynamo

# Make sure custom modules (GroundinDINO, etc.) are registered
from ground_dino.utils import register_all_modules
register_all_modules()


def seed_everything(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train / Eval Grounding DINO')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='dir to save logs and models')
    parser.add_argument('--seed', type=int, default=123, help='Seed value')
    parser.add_argument('--debug', action='store_true')

    # AMP toggle
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable automatic-mixed-precision training')

    # Auto-scale LR
    parser.add_argument('--auto-scale-lr', action='store_true',
                        help='enable automatically scaling LR')

    # Resume
    parser.add_argument('--resume', nargs='?', type=str, const='auto',
                        help=('If a path is given, resume from it. '
                              'If specified with no value, auto-resume from latest.'))

    # Override cfg keys from CLI
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help=('Override settings in cfg, e.g. key=val key="[a,b]" '
                              'key="[(a,b),(c,d)]"'))

    # Launchers
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    # torchrun passes --local-rank or --local_rank depending on version
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    # Eval-only mode
    parser.add_argument('--val-only', action='store_true',
                        help='Run validation/test once and exit (no training).')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def find_ckpts(work_dir: str):
    """Return (best_ckpt, last_ckpt) if present."""
    best = None
    last = None
    # BEST: anything containing 'best' in filename
    best_cands = sorted(glob.glob(osp.join(work_dir, '*best*.pth')))
    if best_cands:
        best = best_cands[-1]
    # LAST: highest epoch_*.pth
    epoch_cands = sorted(glob.glob(osp.join(work_dir, 'epoch_*.pth')))
    if epoch_cands:
        last = epoch_cands[-1]
    return best, last


def run_test_with_ckpt(base_cfg: Config, ckpt_path: str, title: str):
    """Clone cfg, set load_from to ckpt, and run test loop."""
    print_log(f'[{title}] Evaluating checkpoint: {ckpt_path}', logger='current')
    test_cfg = base_cfg.copy()
    test_cfg.resume = False
    test_cfg.load_from = ckpt_path  # ensure we load this checkpoint
    test_runner = Runner.from_cfg(test_cfg)
    test_runner.test()


def main():
    args = parse_args()
    seed_everything(args.seed)
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir: CLI > cfg > default
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # AMP
    if args.amp:
        if getattr(cfg, 'optim_wrapper', None) is None:
            cfg.optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
        else:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # LR auto-scale
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and 'enable' in cfg.auto_scale_lr and 'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Missing auto_scale_lr.enable/base_batch_size in cfg.')

    # Resume handling
    if args.resume == 'auto':
        cfg.resume = True
        # Avoid overriding resume with pretrain
        # (Runner will use last_checkpoint in work_dir)
        cfg.load_from = None
    elif isinstance(args.resume, str):
        # mmengine supports resume as a path string
        cfg.resume = args.resume
    # else: leave cfg.resume/load_from as defined in the config

    # Build runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    if args.debug:
        print_log('Debug flag set; exiting before train/test.', logger='current')
        return

    # -------- VAL-ONLY mode ----------
    if args.val_only:
        print_log('Running validation/test only.', logger='current')
        runner.test()
        return

    # -------- TRAIN ----------
    runner.train()

    # -------- Always run post-train EVALS: BEST and LAST ----------
    try:
        best_ckpt, last_ckpt = find_ckpts(cfg.work_dir)

        if best_ckpt is not None:
            run_test_with_ckpt(cfg, best_ckpt, 'BEST')
        else:
            print_log('No *best*.pth found (ensure save_best in checkpoint hook).',
                      logger='current')

        if last_ckpt is not None:
            run_test_with_ckpt(cfg, last_ckpt, 'LAST')
        else:
            print_log('No epoch_*.pth found for LAST evaluation.', logger='current')

    except Exception as e:
        print_log(f'Post-train evals failed: {e}', logger='current')


if __name__ == '__main__':
    main()
