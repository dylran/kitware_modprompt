# CUDA_VISIBLE_DEVICES=1 python tools/train_yolo.py configs/yolo_w/flir/yolo_world_s_modprompt_resnet34_camel_ir.py \
#   --work-dir output/camel_ir_4cls \

CUDA_VISIBLE_DEVICES=1  python tools/train_yolo.py configs/yolo_w/flir/yolo_world_s_modprompt_resnet34_camel_ir.py \
  --work-dir output/camel_ir_quickcheck \
  --cfg-options train_cfg.max_epochs=1 train_cfg.val_interval=1 train_cfg.dynamic_intervals=[] default_hooks.checkpoint.interval=1
