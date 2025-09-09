#!/usr/bin/env python3
"""
Enhanced YOLO-World Prediction Visualization Tool with Debug Features
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add your project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.apis import init_detector, inference_detector
    from mmdet.registry import VISUALIZERS
    from mmdet.structures import DetDataSample
    import mmcv
    MMDET_AVAILABLE = True
except ImportError as e:
    print(f"MMDetection imports failed: {e}")
    print("Please install mmdetection and dependencies")
    MMDET_AVAILABLE = False


class DebugPredictionVisualizer:
    """Enhanced visualizer with comprehensive debug features."""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        """
        Initialize the prediction visualizer.
        
        Args:
            config_path: Path to your config file (.py)
            checkpoint_path: Path to trained model checkpoint (.pth)
            device: Device to run inference on
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Class information
        self.class_names = [
            'person', 'bike', 'car', 'motor', 'bus', 'train', 'truck',
            'light', 'hydrant', 'sign', 'dog', 'deer', 'skateboard',
            'stroller', 'scooter', 'other vehicle'
        ]
        self.colors = self._generate_colors(len(self.class_names))
        
        # Model will be loaded lazily
        self.model = None
        self.cfg = None
        
        # Debug info
        self.debug_info = {}
        
    def _generate_colors(self, num_classes: int) -> List[Tuple[float, float, float]]:
        """Generate distinct colors for each class."""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(rgb)
        return colors
    
    def debug_paths_and_files(self):
        """Debug function to check all paths and files."""
        print("\n🔍 DEBUG: Checking paths and files...")
        
        # Check config and checkpoint
        print(f"\n1. Model Files:")
        print(f"   Config: {self.config_path}")
        print(f"   ✓ Exists: {os.path.exists(self.config_path)}")
        print(f"   Checkpoint: {self.checkpoint_path}")
        print(f"   ✓ Exists: {os.path.exists(self.checkpoint_path)}")
        
        # Check dataset configs from your config file
        dataset_configs = {
            'val': {
                'ann_file': '/data/ModPrompt/src/new_data/splits/flir_v2/ir/val.kwcoco.json',
                'img_root': '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2/images_thermal_val'
            },
            'test': {
                'ann_file': '/data/ModPrompt/src/new_data/splits/flir_v2/ir/test.kwcoco.json',
                'img_root': '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2/video_thermal_test'
            }
        }
        
        for split, config in dataset_configs.items():
            print(f"\n2. {split.upper()} Dataset:")
            print(f"   Annotation file: {config['ann_file']}")
            print(f"   ✓ Exists: {os.path.exists(config['ann_file'])}")
            print(f"   Image root: {config['img_root']}")
            print(f"   ✓ Exists: {os.path.exists(config['img_root'])}")
            
            # Check if data subdirectory exists
            data_path = os.path.join(config['img_root'], 'data')
            print(f"   Data subdir: {data_path}")
            print(f"   ✓ Exists: {os.path.exists(data_path)}")
            
            # Check annotation file content
            if os.path.exists(config['ann_file']):
                try:
                    with open(config['ann_file'], 'r') as f:
                        data = json.load(f)
                    
                    num_images = len(data.get('images', []))
                    num_annotations = len(data.get('annotations', []))
                    categories = data.get('categories', [])
                    
                    print(f"   📊 Stats: {num_images} images, {num_annotations} annotations")
                    print(f"   📊 Categories: {len(categories)} classes")
                    
                    # Sample some image paths
                    if num_images > 0:
                        sample_images = data['images'][:3]
                        print(f"   📁 Sample image paths:")
                        for img in sample_images:
                            full_path = os.path.join(config['img_root'], 'data', img['file_name'])
                            print(f"      {img['file_name']} -> ✓ {os.path.exists(full_path)}")
                    
                except Exception as e:
                    print(f"   ❌ Error reading annotation file: {e}")
        
        return dataset_configs
    
    def load_model(self):
        """Load the trained YOLO-World model."""
        if not MMDET_AVAILABLE:
            raise ImportError("MMDetection not available. Please install it first.")
            
        print(f"\n🤖 Loading model from config: {self.config_path}")
        print(f"🤖 Loading checkpoint: {self.checkpoint_path}")
        
        # Check if files exist
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        # Load config
        self.cfg = Config.fromfile(self.config_path)

        print("\n[DEBUG] Effective test pipeline:")
        for i, t in enumerate(self.cfg.test_dataloader.dataset.pipeline):
            print(f"  {i:02d}: {t}")

        
        # Debug: Print some config info
        print(f"🔧 Config type: {type(self.cfg)}")
        print(f"🔧 Model type: {self.cfg.get('model', {}).get('type', 'Unknown')}")
        
        # Initialize model
        try:
            self.model = init_detector(
                config=self.cfg,
                checkpoint=self.checkpoint_path,
                device=self.device
            )
            print(f"✅ Model loaded successfully on {self.device}")
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"   📊 Total parameters: {total_params:,}")
            print(f"   📊 Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
    def load_coco_annotations(self, ann_file: str) -> Dict[str, Any]:
        """Load COCO-format annotations with debug info."""
        print(f"📖 Loading annotations from: {ann_file}")
        
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            print(f"   ✅ Loaded {len(data.get('images', []))} images")
            print(f"   ✅ Loaded {len(data.get('annotations', []))} annotations")
            print(f"   ✅ Loaded {len(data.get('categories', []))} categories")
            
            return data
            
        except Exception as e:
            print(f"   ❌ Error loading annotations: {e}")
            raise
        
    def inference_single_image(self, img_path: str, conf_threshold: float = 0.3) -> Dict:
        """
        Run inference on a single image with clean success/error logging.
        Only prints '✅ Inference completed' when no exception occurs.
        """
        if self.model is None:
            self.load_model()

        print(f"🔍 Running inference on: {os.path.basename(img_path)}")
        print(f"   Full path: {img_path}")
        print(f"   File exists: {os.path.exists(img_path)}")

        if not os.path.exists(img_path):
            print(f"   ❌ Image file not found!")
            return {'bboxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])}

        try:
            print(f"   🚀 Running model inference...")
            result = inference_detector(self.model, img_path)

            # --- Parse MMDet result into numpy arrays ---
            # mmdet>=3 returns a DetDataSample with .pred_instances
            pred_instances = result.pred_instances
            raw_bboxes = pred_instances.bboxes.cpu().numpy() if len(pred_instances.bboxes) > 0 else np.array([])
            raw_scores = pred_instances.scores.cpu().numpy() if len(pred_instances.scores) > 0 else np.array([])
            raw_labels = pred_instances.labels.cpu().numpy() if len(pred_instances.labels) > 0 else np.array([])

            predictions = {
                'bboxes': raw_bboxes,
                'scores': raw_scores,
                'labels': raw_labels
            }

            # --- Confidence filtering ---
            if len(predictions['scores']) > 0:
                valid = predictions['scores'] >= conf_threshold
                predictions = {
                    'bboxes': predictions['bboxes'][valid],
                    'scores': predictions['scores'][valid],
                    'labels': predictions['labels'][valid],
                }

            # --- Only here do we print success, because no exception happened ---
            print(f"   ✅ Inference completed: {len(predictions['bboxes'])} detections (conf > {conf_threshold})")
            if len(raw_scores) > 0:
                print(f"   📊 Raw score range: {raw_scores.min():.3f} - {raw_scores.max():.3f}")

            return predictions

        except Exception as e:
            # Any failure (e.g., missing 'texts') is caught here; we DO NOT print a success message.
            print(f"   ❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return {'bboxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])}


    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
        # Convert from [x, y, w, h] to [x1, y1, x2, y2] if needed
        if len(box1) == 4 and len(box2) == 4:
            if box1[2] < box1[0]:  # Check if it's [x, y, w, h] format
                box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
            if box2[2] < box2[0]:
                box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_predictions_to_gt(self, predictions: Dict, ground_truth: List[Dict], iou_threshold: float = 0.5):
        """Match predictions to ground truth annotations."""
        pred_bboxes = predictions['bboxes']
        pred_labels = predictions['labels']
        pred_scores = predictions['scores']
        
        print(f"   🎯 Matching {len(pred_bboxes)} predictions to {len(ground_truth)} GT annotations")
        
        if len(pred_bboxes) == 0:
            return {
                'tp': [],
                'fp': list(range(len(pred_bboxes))),
                'fn': list(range(len(ground_truth))),
                'matched_gt': [],
                'matched_pred': []
            }
        
        # Convert GT format
        gt_bboxes = []
        gt_labels = []
        for ann in ground_truth:
            bbox = ann['bbox']  # [x, y, w, h]
            gt_bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # [x1, y1, x2, y2]
            gt_labels.append(ann['category_id'] - 1)  # Convert to 0-indexed
        
        if len(gt_bboxes) == 0:
            return {
                'tp': [],
                'fp': list(range(len(pred_bboxes))),
                'fn': [],
                'matched_gt': [],
                'matched_pred': []
            }
        
        gt_bboxes = np.array(gt_bboxes)
        gt_labels = np.array(gt_labels)
        
        print(f"   🎯 GT labels: {np.unique(gt_labels)}")
        print(f"   🎯 Pred labels: {np.unique(pred_labels)}")
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))
        for i, pred_bbox in enumerate(pred_bboxes):
            for j, gt_bbox in enumerate(gt_bboxes):
                iou_matrix[i, j] = self.calculate_iou(pred_bbox, gt_bbox)
        
        # Match predictions to ground truth
        tp_indices = []
        fp_indices = []
        matched_gt = set()
        matched_pred = []
        
        # Sort predictions by confidence (highest first)
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        for pred_idx in sorted_indices:
            best_gt_idx = -1
            best_iou = 0.0
            
            for gt_idx in range(len(gt_bboxes)):
                if gt_idx in matched_gt:
                    continue
                    
                iou = iou_matrix[pred_idx, gt_idx]
                if iou > best_iou and iou >= iou_threshold and pred_labels[pred_idx] == gt_labels[gt_idx]:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                tp_indices.append(pred_idx)
                matched_gt.add(best_gt_idx)
                matched_pred.append((pred_idx, best_gt_idx))
            else:
                fp_indices.append(pred_idx)
        
        fn_indices = [i for i in range(len(gt_bboxes)) if i not in matched_gt]
        
        print(f"   📊 Matching results: TP={len(tp_indices)}, FP={len(fp_indices)}, FN={len(fn_indices)}")
        
        return {
            'tp': tp_indices,
            'fp': fp_indices,
            'fn': fn_indices,
            'matched_gt': list(matched_gt),
            'matched_pred': matched_pred
        }
    
    def visualize_prediction_vs_gt(self, 
                                  img_path: str,
                                  ground_truth: List[Dict],
                                  predictions: Dict,
                                  dataset_name: str = "Dataset",
                                  sample_idx: int = 0,
                                  conf_threshold: float = 0.3):
        """
        Create side-by-side visualization of ground truth and predictions.
        """
        print(f"🎨 Creating visualization for sample {sample_idx}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not load image {img_path}")
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"   📐 Image shape: {img_rgb.shape}")
        
        # Match predictions to ground truth
        matching = self.match_predictions_to_gt(predictions, ground_truth)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Ground Truth
        ax1.imshow(img_rgb)
        ax1.set_title(f'{dataset_name} - Ground Truth\n{os.path.basename(img_path)}', fontsize=12)
        
        for i, ann in enumerate(ground_truth):
            bbox = ann['bbox']  # [x, y, w, h]
            x, y, w, h = bbox
            cat_id = ann['category_id'] - 1  # Convert to 0-indexed
            
            if 0 <= cat_id < len(self.class_names):
                class_name = self.class_names[cat_id]
                color = self.colors[cat_id]
            else:
                class_name = f"class_{cat_id}"
                color = (1.0, 0.0, 0.0)
            
            # Color code: Green for matched GT, Orange for unmatched (FN)
            edge_color = 'green' if i in matching['matched_gt'] else 'orange'
            edge_width = 3 if i in matching['matched_gt'] else 2
            
            rect = Rectangle((x, y), w, h, linewidth=edge_width, 
                           edgecolor=edge_color, facecolor='none')
            ax1.add_patch(rect)
            
            # Add label
            ax1.text(x, y-5, class_name, color=edge_color, fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax1.axis('off')
        
        # Plot 2: Predictions
        ax2.imshow(img_rgb)
        ax2.set_title(f'{dataset_name} - Predictions (conf > {conf_threshold})', fontsize=12)
        
        pred_bboxes = predictions['bboxes']
        pred_scores = predictions['scores']
        pred_labels = predictions['labels']
        
        for i in range(len(pred_bboxes)):
            bbox = pred_bboxes[i]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            label_idx = int(pred_labels[i])
            score = pred_scores[i]
            
            if 0 <= label_idx < len(self.class_names):
                class_name = self.class_names[label_idx]
                color = self.colors[label_idx]
            else:
                class_name = f"class_{label_idx}"
                color = (1.0, 0.0, 0.0)
            
            # Color code: Green for TP, Red for FP
            edge_color = 'green' if i in matching['tp'] else 'red'
            edge_width = 3 if i in matching['tp'] else 2
            
            rect = Rectangle((x1, y1), w, h, linewidth=edge_width,
                           edgecolor=edge_color, facecolor='none')
            ax2.add_patch(rect)
            
            # Add label with confidence
            label_text = f"{class_name} ({score:.2f})"
            ax2.text(x1, y1-5, label_text, color=edge_color, fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='True Positive (TP)'),
            Line2D([0], [0], color='red', lw=2, label='False Positive (FP)'),
            Line2D([0], [0], color='orange', lw=2, label='False Negative (FN)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        # Add metrics text
        tp_count = len(matching['tp'])
        fp_count = len(matching['fp'])
        fn_count = len(matching['fn'])
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        
        metrics_text = f"TP: {tp_count}, FP: {fp_count}, FN: {fn_count}\n"
        metrics_text += f"Precision: {precision:.3f}, Recall: {recall:.3f}"
        
        fig.suptitle(metrics_text, fontsize=10, y=0.08)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save the plot
        output_dir = "prediction_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = f"{output_dir}/{dataset_name.lower()}_predictions_sample_{sample_idx}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✅ Saved prediction visualization to: {plot_filename}")
        plt.close()
        
        return {
            'tp': tp_count,
            'fp': fp_count, 
            'fn': fn_count,
            'precision': precision,
            'recall': recall,
            'image_path': img_path
        }
    
    def run_evaluation(self, 
                      num_samples: int = 3,
                      conf_threshold: float = 0.3,
                      splits: List[str] = ['val', 'test'],
                      debug_first_sample: bool = True):
        """
        Run evaluation on validation and test sets with enhanced debugging.
        """
        print(f"\n🚀 Starting evaluation with {num_samples} samples per split")
        print(f"🚀 Confidence threshold: {conf_threshold}")
        print(f"🚀 Splits to evaluate: {splits}")
        
        # First, run path debugging
        dataset_configs = self.debug_paths_and_files()
        
        # Load model
        self.load_model()
        
        all_results = {}
        
        for split in splits:
            if split not in dataset_configs:
                print(f"⚠️  Unknown split '{split}', skipping...")
                continue
                
            print(f"\n{'='*50}")
            print(f"🔍 EVALUATING {split.upper()} SPLIT")
            print(f"{'='*50}")
            
            config = dataset_configs[split]
            ann_file = config['ann_file']
            img_root = config['img_root']
            
            # Check if files exist
            if not os.path.exists(ann_file):
                print(f"❌ Annotation file not found: {ann_file}")
                continue
            if not os.path.exists(img_root):
                print(f"❌ Image root not found: {img_root}")
                continue
            
            # Load annotations
            try:
                coco_data = self.load_coco_annotations(ann_file)
                images = coco_data['images']
                annotations = coco_data['annotations']
                
                print(f"📊 Dataset stats: {len(images)} images, {len(annotations)} annotations")
                
            except Exception as e:
                print(f"❌ Error loading annotations: {e}")
                continue
            
            # Create mapping from image_id to annotations
            img_to_anns = {}
            for ann in annotations:
                img_id = ann['image_id']
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)
            
            print(f"📊 Images with annotations: {len(img_to_anns)}")
            
            # Filter images that have annotations
            images_with_anns = [img for img in images if img['id'] in img_to_anns]
            print(f"📊 Available images with GT: {len(images_with_anns)}")
            
            if len(images_with_anns) == 0:
                print(f"❌ No images with annotations found for {split} split!")
                continue
            
            # Sample random images
            sample_size = min(num_samples, len(images_with_anns))
            sample_images = random.sample(images_with_anns, sample_size)
            
            print(f"🎲 Selected {sample_size} random samples")
            
            split_results = []
            
            for idx, img_info in enumerate(sample_images):
                print(f"\n🖼️  Processing {split} sample {idx + 1}/{len(sample_images)}")
                print(f"   📁 File: {img_info['file_name']}")
                print(f"   🆔 Image ID: {img_info['id']}")
                
                # Build image path - try multiple possible paths
                possible_paths = [
                    os.path.join(img_root, 'data', img_info['file_name']),
                    os.path.join(img_root, img_info['file_name']),
                ]
                
                img_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        img_path = path
                        break
                
                if img_path is None:
                    print(f"❌ Image not found in any of these locations:")
                    for path in possible_paths:
                        print(f"      {path}")
                    continue
                
                print(f"   ✅ Found image at: {img_path}")
                
                # Get ground truth for this image
                img_id = img_info['id']
                gt_annotations = img_to_anns.get(img_id, [])
                print(f"   📊 Ground truth: {len(gt_annotations)} annotations")
                
                # Debug first sample in detail
                if debug_first_sample and idx == 0:
                    print(f"\n🔬 DETAILED DEBUG INFO FOR FIRST SAMPLE:")
                    print(f"   🖼️  Image info: {img_info}")
                    print(f"   📝 GT annotations:")
                    for i, ann in enumerate(gt_annotations):
                        print(f"      {i}: bbox={ann['bbox']}, category_id={ann['category_id']}, area={ann.get('area', 'N/A')}")
                
                # Run inference
                try:
                    predictions = self.inference_single_image(img_path, conf_threshold)
                    print(f"   📊 Predictions: {len(predictions['bboxes'])} detections (conf > {conf_threshold})")
                    
                    if len(predictions['bboxes']) == 0:
                        print(f"   ⚠️  No predictions above confidence threshold!")
                    
                    # Visualize
                    result = self.visualize_prediction_vs_gt(
                        img_path=img_path,
                        ground_truth=gt_annotations,
                        predictions=predictions,
                        dataset_name=split.capitalize(),
                        sample_idx=idx,
                        conf_threshold=conf_threshold
                    )
                    
                    if result:
                        split_results.append(result)
                        print(f"   📊 Metrics - TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
                        print(f"   📊 Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing image: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            all_results[split] = split_results
            print(f"\n✅ {split.upper()} split completed: {len(split_results)} successful evaluations")
        
        # Create summary
        self._create_summary_report(all_results, conf_threshold)
        
        return all_results
    
    def _create_summary_report(self, all_results: Dict, conf_threshold: float):
        """Create a summary report of all results."""
        output_dir = "prediction_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Text report
        report_path = f"{output_dir}/prediction_report.txt"
        with open(report_path, 'w') as f:
            f.write("YOLO-World Prediction Evaluation Report (Enhanced Debug)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {self.checkpoint_path}\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Confidence Threshold: {conf_threshold}\n\n")
            
            for split, results in all_results.items():
                f.write(f"{split.upper()} Split Results:\n")
                f.write("-" * 30 + "\n")
                
                if not results:
                    f.write("❌ No results available - check debug output above.\n\n")
                    continue
                
                # Aggregate metrics
                total_tp = sum(r['tp'] for r in results)
                total_fp = sum(r['fp'] for r in results)
                total_fn = sum(r['fn'] for r in results)
                
                avg_precision = np.mean([r['precision'] for r in results])
                avg_recall = np.mean([r['recall'] for r in results])
                
                f.write(f"✅ Samples evaluated: {len(results)}\n")
                f.write(f"📊 Total TP: {total_tp}\n")
                f.write(f"📊 Total FP: {total_fp}\n")
                f.write(f"📊 Total FN: {total_fn}\n")
                f.write(f"📊 Average Precision: {avg_precision:.3f}\n")
                f.write(f"📊 Average Recall: {avg_recall:.3f}\n")
                
                if avg_precision + avg_recall > 0:
                    f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
                    f.write(f"📊 Average F1 Score: {f1_score:.3f}\n")
                
                f.write("\nPer-sample results:\n")
                for i, result in enumerate(results):
                    f.write(f"  Sample {i}: {os.path.basename(result['image_path'])} - "
                           f"P: {result['precision']:.3f}, R: {result['recall']:.3f}\n")
                f.write("\n")
        
        print(f"📄 Saved evaluation report to: {report_path}")

    def quick_debug_test(self, split: str = 'val', max_samples: int = 1):
        """Quick debug test to check if everything works with minimal samples."""
        print(f"\n🧪 QUICK DEBUG TEST - {split.upper()} split")
        print("=" * 50)
        
        # Check paths first
        dataset_configs = self.debug_paths_and_files()
        
        if split not in dataset_configs:
            print(f"❌ Split {split} not in dataset_configs")
            return
        
        config = dataset_configs[split]
        ann_file = config['ann_file']
        img_root = config['img_root']
        
        if not os.path.exists(ann_file):
            print(f"❌ Annotation file missing: {ann_file}")
            return
        
        if not os.path.exists(img_root):
            print(f"❌ Image root missing: {img_root}")
            return
        
        # Load annotations
        try:
            coco_data = self.load_coco_annotations(ann_file)
            images = coco_data['images']
            annotations = coco_data['annotations']
            
        except Exception as e:
            print(f"❌ Failed to load annotations: {e}")
            return
        
        # Create image-to-annotations mapping
        img_to_anns = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Find images with annotations
        images_with_anns = [img for img in images if img['id'] in img_to_anns]
        
        if len(images_with_anns) == 0:
            print(f"❌ No images with annotations found!")
            return
        
        print(f"✅ Found {len(images_with_anns)} images with annotations")
        
        # Test first few images
        test_samples = min(max_samples, len(images_with_anns))
        print(f"🧪 Testing {test_samples} samples...")
        
        # Load model
        try:
            self.load_model()
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return
        
        for i in range(test_samples):
            img_info = images_with_anns[i]
            print(f"\n🧪 Test sample {i+1}: {img_info['file_name']}")
            
            # Try to find the image
            possible_paths = [
                os.path.join(img_root, 'data', img_info['file_name']),
                os.path.join(img_root, img_info['file_name']),
            ]
            
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    break
            
            if img_path is None:
                print(f"   ❌ Image not found")
                continue
                
            print(f"   ✅ Image found: {img_path}")
            
            # Get GT
            gt_annotations = img_to_anns.get(img_info['id'], [])
            print(f"   📊 GT annotations: {len(gt_annotations)}")
            
            # Test inference
            try:
                predictions = self.inference_single_image(img_path, 0.1)  # Low threshold for testing
                print(f"   ✅ Inference successful: {len(predictions['bboxes'])} detections")
                
                # Try visualization
                result = self.visualize_prediction_vs_gt(
                    img_path=img_path,
                    ground_truth=gt_annotations,
                    predictions=predictions,
                    dataset_name=f"Debug_{split}",
                    sample_idx=i,
                    conf_threshold=0.1
                )
                
                if result:
                    print(f"   ✅ Visualization successful")
                    print(f"   📊 TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
                else:
                    print(f"   ❌ Visualization failed")
                    
            except Exception as e:
                print(f"   ❌ Inference/visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Quick debug test completed!")


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(description='Enhanced YOLO-World Prediction Visualizer with Debug')
    parser.add_argument('--config', type=str, 
                       default='configs/yolo_w/flir/yolo_world_s_modprompt_resnet34_flirv2_ir.py',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                       default='/data/ModPrompt/src/output/flir_v2_16cls/best_coco_bbox_mAP_epoch_5.pth', 
                       help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run inference on')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to visualize per split')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Confidence threshold for predictions')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                       help='Which splits to evaluate')
    parser.add_argument('--debug-only', action='store_true',
                       help='Run quick debug test only')
    parser.add_argument('--debug-split', type=str, default='val',
                       help='Which split to use for debug test')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced YOLO-World Prediction Visualizer")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    
    # Initialize visualizer
    visualizer = DebugPredictionVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    if args.debug_only:
        print(f"\n🧪 Running debug test only on {args.debug_split} split...")
        visualizer.quick_debug_test(split=args.debug_split, max_samples=2)
    else:
        print(f"Confidence threshold: {args.conf_threshold}")
        print(f"Samples per split: {args.num_samples}")
        print(f"Splits to evaluate: {args.splits}")
        
        # Run full evaluation
        results = visualizer.run_evaluation(
            num_samples=args.num_samples,
            conf_threshold=args.conf_threshold,
            splits=args.splits,
            debug_first_sample=True
        )
        
        print("\n" + "=" * 60)
        print("🎉 Evaluation complete!")
        print("\n📁 Generated files:")
        print("  - prediction_visualizations/*_predictions_sample_*.png")
        print("  - prediction_visualizations/prediction_report.txt")
        
        # Summary
        total_successful = sum(len(results[split]) for split in results)
        print(f"\n📊 Total successful evaluations: {total_successful}")
        
        if total_successful == 0:
            print("⚠️  No successful evaluations! Check the debug output above.")
            print("💡 Try running with --debug-only flag first to diagnose issues.")


if __name__ == '__main__':
    main()