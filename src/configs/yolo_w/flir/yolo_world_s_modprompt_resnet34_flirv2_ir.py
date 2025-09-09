# yolo_world_s_modprompt_resnet34_flirv2_ir.py
_base_ = 'flir_base.py'  # same folder

custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# ====================== Classes / text prompts ======================
num_classes = 16
num_training_classes = 16
metainfo = dict(classes=[
    'person', 'bike', 'car', 'motor', 'bus', 'train', 'truck',
    'light', 'hydrant', 'sign', 'dog', 'deer', 'skateboard',
    'stroller', 'scooter', 'other vehicle'
])
CLASS_TEXT_PATH = 'data/texts/flir_v2_16_classes.json'

from pathlib import Path
import json

CLASS_TEXTS = json.loads(Path(CLASS_TEXT_PATH).read_text())
# accept ["person", ...] or [["person"], ...]
if len(CLASS_TEXTS) > 0 and isinstance(CLASS_TEXTS[0], str):
    CLASS_TEXTS = [[c] for c in CLASS_TEXTS]

# ====================== Hyper-params ======================
max_epochs = 80
close_mosaic_epochs = 10
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 8
load_from = 'pretrained_models/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained.pth'

# ====================== KWCOCO splits ======================
SPLIT_ROOT = '/data/ModPrompt/src/new_data/splits/flir_v2/ir'
TRAIN_JSON = f'{SPLIT_ROOT}/train.kwcoco.json'
VAL_JSON   = f'{SPLIT_ROOT}/val.kwcoco.json'
TEST_JSON  = f'{SPLIT_ROOT}/test.kwcoco.json'

# ====================== Image roots per split ======================
# Your file_name values are basenames; actual images live under .../data
TRAIN_IMG_ROOT = '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2/images_thermal_train'
VAL_IMG_ROOT   = '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2/images_thermal_val'
TEST_IMG_ROOT  = '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2/video_thermal_test'
IMG_PREFIX = 'data'  # os.path.join(root, 'data', file_name)

# ====================== Model ======================
model = dict(
    type='YOLOWorldDetector',
    img_prompt='modprompt',
    modprompt_backbone='resnet34',
    modprompt_encoder_depth=5,
    modprompt_in_channels=3,
    modprompt_out_channels=3,
    modprompt_encoder_weights='imagenet',
    modprompt_alpha=1.0,
    prompt_size=30,
    image_size_width=_base_.img_scale[0],
    image_size_height=_base_.img_scale[1],
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model=_base_.model.backbone,
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        )
    ),
    neck=dict(
        type='YOLOWorldDualPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        text_enhancder=dict(type='ImagePoolingAttentionModule',
                            embed_channels=256, num_heads=8)
    ),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            embed_dims=text_channels,
            num_classes=num_training_classes
        )
    ),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes))
)

# ====================== Pipelines ======================
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(type='YOLOv5RandomAffine',
         max_rotate_degree=0.0,
         max_shear_degree=0.0,
         scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
         max_aspect_ratio=_base_.max_aspect_ratio,
         border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
         border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]

test_pipeline = [
    # Put this FIRST so texts exist before any other text-aware step.
    dict(
        type='RandomLoadText',
        class_texts=CLASS_TEXTS,           # <-- pass data, not a file handle
        num_neg_samples=(0, 0),
        max_num_samples=num_training_classes,
        padding_to_max=True,
        padding_value=''
    ),
    *_base_.test_pipeline[:-1],
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id','img_path','ori_shape','img_shape',
                   'scale_factor','pad_param','texts')
    ),
]

# ====================== Datasets ======================
flirv2_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        metainfo=metainfo,
        type='YOLOv5CocoDataset',
        data_root=TRAIN_IMG_ROOT,
        ann_file=TRAIN_JSON,
        data_prefix=dict(img=IMG_PREFIX),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)
    ),
    class_text_path=CLASS_TEXT_PATH,
    pipeline=train_pipeline
)
flirv2_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=metainfo,
        data_root=VAL_IMG_ROOT,
        test_mode=True,
        ann_file=VAL_JSON,
        data_prefix=dict(img=IMG_PREFIX),
        batch_shapes_cfg=None
    ),
    class_text_path=CLASS_TEXT_PATH,
    pipeline=test_pipeline
)
flirv2_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=metainfo,
        data_root=TEST_IMG_ROOT,
        test_mode=True,
        ann_file=TEST_JSON,
        data_prefix=dict(img=IMG_PREFIX),
        batch_shapes_cfg=None
    ),
    class_text_path=CLASS_TEXT_PATH,
    pipeline=test_pipeline
)

# ====================== Dataloaders ======================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=flirv2_train_dataset
)
val_dataloader = dict(dataset=flirv2_val_dataset)
test_dataloader = dict(dataset=flirv2_test_dataset)

# ====================== Evaluators ======================
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=VAL_JSON,
    metric='bbox'
)
test_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=TEST_JSON,
    metric='bbox'
)

# ====================== Training loop / hooks / optim ======================
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=save_epoch_intervals, save_best='auto', rule='greater')
)
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001,
         update_buffers=True, strict_load=False, priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), _base_.val_interval_stage2)]
)
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=base_lr,
                   weight_decay=weight_decay, batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0,
                       custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                                    'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor'
)
