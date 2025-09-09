# g_dino_flir_v2_ir.py
# Grounding-DINO + ModPrompt for FLIR_V2 (IR), COCO-style splits

_base_ = [
    '../_base_/coco_detection.py',
    '../_base_/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# --- Ensure Grounding-DINO is registered ---
custom_imports = dict(
    imports=['ground_dino.models'],  # this package's __init__ imports detectors/*
    allow_failed_imports=False
)

# --- Checkpoint / run control ---
# You can switch to pretrain-on-Objects365 if you prefer
# load_from = '/data/ModPrompt/src/pretrained_models/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth'
load_from = '/data/ModPrompt/src/pretrained_models/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth'
resume = False
lang_model_name = 'bert-base-uncased'

# ====================== Split selectors (COCO-style JSON) ======================
DATASET_NAME = 'flir_v2'
MODALITY = 'ir'

SPLIT_ROOT = f'/data/ModPrompt/src/new_data/splits/{DATASET_NAME}/{MODALITY}'
TRAIN_JSON = f'{SPLIT_ROOT}/train.kwcoco.json'
VAL_JSON   = f'{SPLIT_ROOT}/val.kwcoco.json'
TEST_JSON  = f'{SPLIT_ROOT}/test.kwcoco.json'

# Your images live here; JSON "file_name" is a BASENAME (e.g., img.jpg)
DATA_ROOT_FOR_COCO = '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2'
TRAIN_IMG_PREFIX = 'images_thermal_train/data'
VAL_IMG_PREFIX   = 'images_thermal_val/data'
TEST_IMG_PREFIX  = 'video_thermal_test/data'

# ====================== Classes (order matters) ======================
class_name = (
    'person', 'bike', 'car', 'motor', 'bus', 'train', 'truck',
    'light', 'hydrant', 'sign', 'dog', 'deer', 'skateboard',
    'stroller', 'scooter', 'other vehicle'
)
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# ====================== Model ======================
model = dict(
    type='GroundinDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),

    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),

    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=False,
        init_cfg=None,  # disable separate backbone init; we load full model via `load_from`
    ),

    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4
    ),

    encoder=dict(
        num_layers=6,
        num_cp=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)
        ),
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)
        ),
        fusion_layer_cfg=dict(
            v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4
        ),
    ),

    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)
        ),
        post_norm_cfg=None
    ),

    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20
    ),

    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale=0.0, bias=False),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),

    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)
    ),

    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ]
        )
    ),
    test_cfg=dict(max_per_img=300),

    # ----- ModPrompt -----
    fft=0, hft=1, img_prompt='modprompt',
    modprompt_backbone='resnet34',
    modprompt_encoder_depth=5,
    modprompt_in_channels=3,      # If your JPGs are single-channel, set this to 1
    modprompt_out_channels=3,
    modprompt_encoder_weights='imagenet',
    modprompt_alpha=1.0,
    prompt_size=30,
    image_size_width=640,
    image_size_height=512
)

# ====================== Pipelines ======================
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FixScaleResize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='FixScaleResize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')
    )
]

# ====================== Datasets (COCO) ======================
flir_train_dataset = dict(
    _delete_=True,
    type='CocoDataset',
    metainfo=metainfo,
    data_root=DATA_ROOT_FOR_COCO,
    ann_file=TRAIN_JSON,
    data_prefix=dict(img=TRAIN_IMG_PREFIX),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    return_classes=True
)

flir_val_dataset = dict(
    _delete_=True,
    type='CocoDataset',
    metainfo=metainfo,
    data_root=DATA_ROOT_FOR_COCO,
    test_mode=True,
    ann_file=VAL_JSON,
    data_prefix=dict(img=VAL_IMG_PREFIX),
    pipeline=test_pipeline,
    return_classes=True
)

flir_test_dataset = dict(
    _delete_=True,
    type='CocoDataset',
    metainfo=metainfo,
    data_root=DATA_ROOT_FOR_COCO,
    test_mode=True,
    ann_file=TEST_JSON,
    data_prefix=dict(img=TEST_IMG_PREFIX),
    pipeline=test_pipeline,
    return_classes=True
)

# ====================== Dataloaders ======================
train_dataloader = dict(
    batch_size=8,
    dataset=flir_train_dataset
)

val_dataloader = dict(
    batch_size=8,
    dataset=flir_val_dataset
)

test_dataloader = dict(
    batch_size=8,
    dataset=flir_test_dataset
)

# ====================== Evaluators ======================
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    ann_file=VAL_JSON,
    metric='bbox'
)

test_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    ann_file=TEST_JSON,
    metric='bbox'
)

# ====================== Optim / Sched / Hooks ======================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.1)
        }
    )
)

max_epochs = 60  # set to 1 for a quick smoke test
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5)
)
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1
    )
]

# Auto LR scaling assumes base batch size across GPUs
auto_scale_lr = dict(base_batch_size=32)

# (optional flags carried through)
fft = 0
hft = 1
