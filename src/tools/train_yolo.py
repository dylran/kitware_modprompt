# RemovePoliceCVPR
import pdb

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import logging
# import os
import os.path as osp
import re, glob  # <-- added

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--exp-name', help='Additional Name for the experiment')
    parser.add_argument('--seed', type=int, default=123, help='Seed value')
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # parser.add_argument('--val-only', action='store_true',
    #                 help='Run validation once without training')

    return args


def main():
    # pdb.set_trace()
    args = parse_args()
    seed_everything(args.seed)
    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    
    # Link Removeds://mmengine.RemoveLinkCVPRio/en/stable/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend
    # cfg.visualizer = dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')])

    dataset_name = args.config.split('/')[-2] # from the configs/dataset/*.py
    
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if args.config.startswith('projects/'):
            config = args.config[len('projects/'):]
            config = config.replace('/configs/', '/')
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(config)[0])
        else:
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])
        if args.exp_name is not None:
            cfg.work_dir = cfg.work_dir + '_' +  str(args.exp_name)

    # add the name of the dataset at the begin of the work dir
    tmp = cfg.work_dir.split('/')
    tmp[-1] = dataset_name + '_' +  tmp[-1]
    cfg.work_dir = '/'.join(tmp)

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optimWrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None # Removed due to the CVPR link policy
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = None # Removed due to the CVPR link policy

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
        
        
    ## In prompt tuning baseline train just the embeddings
    if "ft" in args.config:
        for name, param in runner.model.named_parameters():
            param.requires_grad = False
            if "backbone" in name or "neck" in name or "bbox_head" in name:
                param.requires_grad = True


    ## In prompt tuning baseline train just the embeddings
    if "head_ft" in args.config:
        for name, param in runner.model.named_parameters():
            param.requires_grad = False
            if "bbox_head" in name:
                param.requires_grad = True


    ## In prompt tuning baseline train just the embeddings
    if "prompt_tuning" in args.config:
        for name, param in runner.model.named_parameters():
            param.requires_grad = False
            if "embeddings" in name:
                param.requires_grad = True

   
    ## check if it is a image prompt training
    if 'img_prompt' in cfg.model.keys():
        for name, param in runner.model.named_parameters():
            param.requires_grad = False
            if "img_prompt" in name:
                param.requires_grad = True
    
    ## task residuals only
    if "task_residuals" in args.config:
        for name, param in runner.model.named_parameters():
            param.requires_grad = False
            if "text_feature_residuals" in name:
                param.requires_grad = True

    ## txt prompt tuning and img prompt
    if "prompt_tuning" in args.config and 'img_prompt' in cfg.model.keys():
        for name, param in runner.model.named_parameters():
            param.requires_grad = False
            if "embeddings" in name:
                param.requires_grad = True
            if "img_prompt" in name:
                param.requires_grad = True

    ## task residuals and img prompt
    if "task_residuals" in args.config and 'img_prompt' in cfg.model.keys():
        for name, param in runner.model.named_parameters():
            param.requires_grad = False
            if "text_feature_residuals" in name:
                param.requires_grad = True
            if "img_prompt" in name:
                param.requires_grad = True
                

    if(args.debug):
        exit()

    # start training
    runner.train()

    # ===================== Run TWO evaluations: BEST and LAST =====================
    try:
        # Only trigger if the model actually has parameters containing "img_prompt"
        has_img_prompt = any(re.search(r'img_prompt', n) for n, _ in runner.model.named_parameters())
        if not has_img_prompt:
            print_log('Skip auto-eval: model has no parameters matching "img_prompt".',
                      logger='current', level=logging.WARNING)
            return

        def pick_best_and_last(work_dir: str):
            best = None
            last = None
            # pick BEST by filename containing 'best'
            best_cands = sorted(glob.glob(osp.join(work_dir, '*best*.pth')))
            if best_cands:
                best = best_cands[-1]
            # pick LAST by highest epoch_*.pth
            epoch_cands = sorted(glob.glob(osp.join(work_dir, 'epoch_*.pth')))
            if epoch_cands:
                last = epoch_cands[-1]
            return best, last

        best_ckpt, last_ckpt = pick_best_and_last(cfg.work_dir)

        # Helper to run one test pass
        def run_test_with_ckpt(title: str, ckpt_path: str):
            print_log(f'[{title}] Evaluating checkpoint: {ckpt_path}', logger='current')
            test_cfg = cfg.copy()
            test_cfg.resume = False
            test_cfg.load_from = ckpt_path
            test_runner = Runner.from_cfg(test_cfg)
            test_runner.test()

        if best_ckpt is not None:
            run_test_with_ckpt('BEST', best_ckpt)
        else:
            print_log('No *best*.pth found. (Enable save_best in your checkpoint hook to create one.)',
                      logger='current', level=logging.WARNING)

        if last_ckpt is not None:
            run_test_with_ckpt('LAST', last_ckpt)
        else:
            print_log('No epoch_*.pth checkpoints found to evaluate LAST.',
                      logger='current', level=logging.WARNING)

    except Exception as e:
        print_log(f'Auto-evals failed: {e}', logger='current', level=logging.ERROR)


if __name__ == '__main__':
    main()
