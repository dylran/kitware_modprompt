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

# launch DDP
# torchrun --nproc_per_node=2 -m tools.train_gd \
#   configs/g_dino/FLIR/flir_modprompt_resnet.py \
#   --work-dir output_gdino_2 \
#   --launcher pytorch

# launch single node with kitware splits config
# CUDA_VISIBLE_DEVICES=1 python -m tools.train_gd \
#   configs/g_dino/FLIR/flir_modprompt_resnet.py \
#   --work-dir output_gdino/flir_aligned_coco_pretrain \

# launch single node with original config
CUDA_VISIBLE_DEVICES=0 python -m tools.train_gd \
  configs/g_dino/FLIR/flir_modprompt_resnet_legacy.py \
  --work-dir output_gdino/flir_aligned_legacy \
