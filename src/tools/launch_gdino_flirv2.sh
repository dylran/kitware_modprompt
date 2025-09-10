# #!/usr/bin/env bash
# # train_flir_v2_ir.sh — single-GPU run for Grounding-DINO + ModPrompt (FLIR V2 IR)

# set -euo pipefail

# # 1) Make project importable
# export PYTHONPATH=/data/ModPrompt/src:$PYTHONPATH

# # 2) Pick the GPU (single GPU)
# export CUDA_VISIBLE_DEVICES=0

# # 3) Config + work dir
# CFG=/data/ModPrompt/src/configs/g_dino/FLIR/flir_v2_modprompt_resnet.py
# WORK=output_gdino/flir_v2_ir

# # 4) Ensure we don't resume accidentally (so `load_from` is honored)
# rm -f "$WORK/last_checkpoint" 2>/dev/null || true

# # 5) Launch single-GPU training
# python -m tools.train_gd \
#   "$CFG" \
#   --work-dir "$WORK" \
#   --seed 42 \
#   --deterministic \
#   --amp
#   # If you want to tweak on the fly:
#   # --cfg-options train_dataloader.batch_size=8 \
#   #               optim_wrapper.optimizer.lr=2e-4 \
#   #               train_cfg.val_interval=5

# echo "✅ Training started with config: $CFG"
# echo "📂 Logs & ckpts -> $WORK"


# # ensure your project is importable
# export PYTHONPATH=/data/ModPrompt/src:$PYTHONPATH

# # make sure resume doesn't override load_from
# rm -f output_gdino_2/last_checkpoint 2>/dev/null

# # launch
# # python -m tools.train_gd configs/g_dino/FLIR/flir_modprompt_resnet.py \
# #   --work-dir output_gdino_2

# torchrun --nproc_per_node=2 tools.train_gd configs/g_dino/FLIR/flir_modprompt_resnet.py \
#   --work-dir output_gdino_2 --launcher pytorch


# tools/launch_gdino.sh

# make project importable
export PYTHONPATH=/data/ModPrompt/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1   # pick your GPUs

# avoid resuming (so load_from is used)
rm -f output_gdino_2/last_checkpoint 2>/dev/null


# CUDA_VISIBLE_DEVICES=1 python tools/train_gd.py \
#   configs/g_dino/FLIR/flir_v2_modprompt_resnet.py \
#   --amp --work-dir output_gdino/flir_gd_amp_flirv2

torchrun --nproc_per_node=2 tools/train_gd.py \
  configs/g_dino/FLIR/flir_v2_modprompt_resnet.py \
  --work-dir output_gdino/flir_v2_ir_ddp \
  --amp \
  --launcher pytorch