# python tools/train_yolo.py configs/yolo_w/flir/yolo_world_s_modprompt_resnet34_flirv2_ir.py \
#   --work-dir output/quick_fix_check \
#   --cfg-options \
#     train_cfg.max_epochs=1 \
#     train_cfg.val_interval=1 \
#     train_cfg.dynamic_intervals=[] \
#     default_hooks.checkpoint.interval=1 \
#     custom_hooks.1.switch_epoch=0

CUDA_VISIBLE_DEVICES=1 python tools/train_yolo.py configs/yolo_w/flir/yolo_world_s_modprompt_resnet34_flirv2_ir.py \
  --work-dir output/v2_16cls
