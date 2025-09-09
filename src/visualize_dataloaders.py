#!/usr/bin/env python3
"""
Dataloader Visualization Script for YOLO-World Config
This script helps verify that your train/val/test dataloaders are working correctly
by visualizing samples from each dataset with their annotations and text prompts.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any

# If you have mmdetection installed, uncomment these:
# from mmdet.apis import init_detector
# from mmengine import Config
# from mmdet.datasets import build_dataset
# from mmdet.datasets.transforms import Compose

class DataloaderVisualizer:
    """Visualizes samples from YOLO-World dataloaders to verify correctness."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the visualizer.
        
        Args:
            config_path: Path to your config file (optional)
        """
        self.config_path = config_path
        self.class_names = [
            'person', 'bike', 'car', 'motor', 'bus', 'train', 'truck',
            'light', 'hydrant', 'sign', 'dog', 'deer', 'skateboard',
            'stroller', 'scooter', 'other vehicle'
        ]
        self.colors = self._generate_colors(len(self.class_names))
        
    def _generate_colors(self, num_classes: int) -> List[Tuple[float, float, float]]:
        """Generate distinct colors for each class."""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            # Convert HSV to RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(rgb)
        return colors
    
    def load_coco_annotations(self, ann_file: str) -> Dict[str, Any]:
        """Load COCO-format annotations."""
        with open(ann_file, 'r') as f:
            data = json.load(f)
        return data
    
    def visualize_sample_basic(self, 
                              img_root: str, 
                              ann_file: str, 
                              num_samples: int = 3,
                              dataset_name: str = "Dataset"):
        """
        Basic visualization without mmdetection dependencies.
        
        Args:
            img_root: Root directory containing images
            ann_file: Path to COCO annotation file  
            num_samples: Number of samples to visualize
            dataset_name: Name of the dataset (train/val/test)
        """
        print(f"\n=== Visualizing {dataset_name} Dataset ===")
        
        # Load annotations
        try:
            coco_data = self.load_coco_annotations(ann_file)
        except Exception as e:
            print(f"Error loading annotations from {ann_file}: {e}")
            return None
            
        images = coco_data['images']
        annotations = coco_data['annotations']
        categories = coco_data.get('categories', [])
        
        # Create mapping from image_id to annotations
        img_to_anns = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Create category mapping
        cat_id_to_name = {}
        for cat in categories:
            cat_id_to_name[cat['id']] = cat['name']
        
        # Sample random images
        sample_images = random.sample(images, min(num_samples, len(images)))
        
        fig, axes = plt.subplots(1, len(sample_images), figsize=(5*len(sample_images), 5))
        if len(sample_images) == 1:
            axes = [axes]
            
        for idx, img_info in enumerate(sample_images):
            ax = axes[idx]
            
            # Load image
            img_path = os.path.join(img_root, 'data', img_info['file_name'])
            print(f"Loading image: {img_path}")
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                ax.text(0.5, 0.5, f"Image not found:\n{img_info['file_name']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{dataset_name} - Image Not Found")
                continue
                
            # Load image (handle both grayscale thermal and RGB)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
                
            # Convert BGR to RGB for matplotlib
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            
            # Draw annotations
            img_id = img_info['id']
            if img_id in img_to_anns:
                for ann in img_to_anns[img_id]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Get category info
                    cat_id = ann['category_id']
                    cat_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")
                    
                    # Use class color if available
                    if cat_name in self.class_names:
                        color_idx = self.class_names.index(cat_name)
                        color = self.colors[color_idx]
                    else:
                        color = (1.0, 0.0, 0.0)  # Red for unknown classes
                    
                    # Draw bounding box
                    rect = Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(x, y-5, cat_name, color=color, fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
            
            ax.set_title(f"{dataset_name} - {img_info['file_name']}")
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save the plot instead of showing it
        output_dir = "dataloader_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = f"{output_dir}/{dataset_name.lower()}_samples.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {plot_filename}")
        plt.close()  # Close the figure to free memory
        
        # Print dataset statistics
        print(f"\n{dataset_name} Dataset Statistics:")
        print(f"  - Total images: {len(images)}")
        print(f"  - Total annotations: {len(annotations)}")
        print(f"  - Categories: {len(categories)}")
        
        # Print class distribution
        class_counts = {}
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")
            class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
        
        print(f"  - Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count}")
        
        # Return statistics for summary plot
        return {
            'name': dataset_name,
            'num_images': len(images),
            'num_annotations': len(annotations),
            'num_categories': len(categories),
            'class_counts': class_counts
        }
    
    def check_file_paths(self, config_dict: Dict[str, Any]):
        """Check if all file paths in the config exist."""
        print("\n=== Checking File Paths ===")
        
        paths_to_check = [
            ("TRAIN_JSON", config_dict.get('TRAIN_JSON')),
            ("VAL_JSON", config_dict.get('VAL_JSON')), 
            ("TEST_JSON", config_dict.get('TEST_JSON')),
            ("TRAIN_IMG_ROOT", config_dict.get('TRAIN_IMG_ROOT')),
            ("VAL_IMG_ROOT", config_dict.get('VAL_IMG_ROOT')),
            ("TEST_IMG_ROOT", config_dict.get('TEST_IMG_ROOT')),
            ("CLASS_TEXT_PATH", config_dict.get('CLASS_TEXT_PATH')),
            ("load_from", config_dict.get('load_from'))
        ]
        
        for name, path in paths_to_check:
            if path:
                exists = os.path.exists(path)
                status = "✓" if exists else "✗"
                print(f"  {status} {name}: {path}")
                if not exists and name.endswith('_ROOT'):
                    # Check if parent directory exists
                    parent = os.path.dirname(path)
                    if os.path.exists(parent):
                        print(f"    (Parent directory exists: {parent})")
            else:
                print(f"  ? {name}: Not specified")

    def create_summary_plot(self, splits_stats: List[Dict]):
        """Create a summary plot comparing all splits."""
        if not splits_stats:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Dataset sizes
        ax1 = axes[0, 0]
        split_names = [s['name'] for s in splits_stats]
        image_counts = [s['num_images'] for s in splits_stats]
        ann_counts = [s['num_annotations'] for s in splits_stats]
        
        x = np.arange(len(split_names))
        width = 0.35
        
        ax1.bar(x - width/2, image_counts, width, label='Images', alpha=0.8)
        ax1.bar(x + width/2, ann_counts, width, label='Annotations', alpha=0.8)
        ax1.set_xlabel('Dataset Split')
        ax1.set_ylabel('Count')
        ax1.set_title('Dataset Sizes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(split_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Class distribution across all splits
        ax2 = axes[0, 1]
        all_classes = set()
        for stats in splits_stats:
            all_classes.update(stats['class_counts'].keys())
        
        all_classes = sorted(list(all_classes))
        split_class_counts = []
        
        for split_stats in splits_stats:
            counts = [split_stats['class_counts'].get(cls, 0) for cls in all_classes]
            split_class_counts.append(counts)
        
        x = np.arange(len(all_classes))
        width = 0.25
        
        for i, (split_name, counts) in enumerate(zip(split_names, split_class_counts)):
            ax2.bar(x + i*width, counts, width, label=split_name, alpha=0.8)
        
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Count')
        ax2.set_title('Class Distribution Across Splits')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(all_classes, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Annotations per image ratio
        ax3 = axes[1, 0]
        ann_per_img = [s['num_annotations'] / max(s['num_images'], 1) for s in splits_stats]
        bars = ax3.bar(split_names, ann_per_img, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon'])
        ax3.set_xlabel('Dataset Split')
        ax3.set_ylabel('Annotations per Image')
        ax3.set_title('Annotation Density')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, ann_per_img):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 4: Top 5 classes by count
        ax4 = axes[1, 1]
        
        # Combine all class counts
        combined_counts = {}
        for stats in splits_stats:
            for cls, count in stats['class_counts'].items():
                combined_counts[cls] = combined_counts.get(cls, 0) + count
        
        # Get top 5 classes
        top_classes = sorted(combined_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        classes, counts = zip(*top_classes)
        ax4.barh(classes, counts, alpha=0.8, color='lightcoral')
        ax4.set_xlabel('Total Count')
        ax4.set_title('Top 5 Most Frequent Classes')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, count in enumerate(counts):
            ax4.text(count + max(counts)*0.01, i, str(count), va='center')
        
        plt.tight_layout()
        
        # Save the summary plot
        output_dir = "dataloader_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        summary_filename = f"{output_dir}/dataset_summary.png"
        plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved dataset summary to: {summary_filename}")
        plt.close()
        
        return summary_filename
    
    def visualize_all_splits(self):
        """Visualize samples from all dataset splits."""
        # Your config values (extracted from the config file)
        config_dict = {
            'TRAIN_JSON': '/data/ModPrompt/src/new_data/splits/flir_v2/ir/train.kwcoco.json',
            'VAL_JSON': '/data/ModPrompt/src/new_data/splits/flir_v2/ir/val.kwcoco.json',
            'TEST_JSON': '/data/ModPrompt/src/new_data/splits/flir_v2/ir/test.kwcoco.json',
            'TRAIN_IMG_ROOT': '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2/images_thermal_train',
            'VAL_IMG_ROOT': '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2/images_thermal_val', 
            'TEST_IMG_ROOT': '/data/ModPrompt/src/data/FLIR_V2/FLIR_ADAS_v2/video_thermal_test',
            'CLASS_TEXT_PATH': 'data/texts/flir_v2_16_classes.json',
            'load_from': 'pretrained_models/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained.pth'
        }
        
        # Check file paths first
        self.check_file_paths(config_dict)
        
        # Store statistics for summary plot
        splits_stats = []
        
        # Visualize each split
        splits = [
            ("Train", config_dict['TRAIN_IMG_ROOT'], config_dict['TRAIN_JSON']),
            ("Validation", config_dict['VAL_IMG_ROOT'], config_dict['VAL_JSON']),
            ("Test", config_dict['TEST_IMG_ROOT'], config_dict['TEST_JSON'])
        ]
        
        for split_name, img_root, ann_file in splits:
            if os.path.exists(ann_file) and os.path.exists(img_root):
                stats = self.visualize_sample_basic(img_root, ann_file, num_samples=3, 
                                                  dataset_name=split_name)
                if stats:
                    splits_stats.append(stats)
            else:
                print(f"\nSkipping {split_name} split - missing files")
        
        # Create summary plot
        if splits_stats:
            self.create_summary_plot(splits_stats)
    
    def check_class_text_file(self, class_text_path: str):
        """Check the class text file used for text prompts."""
        print(f"\n=== Checking Class Text File ===")
        
        if not os.path.exists(class_text_path):
            print(f"✗ Class text file not found: {class_text_path}")
            print("Creating a sample class text file...")
            
            # Create sample class text file
            os.makedirs(os.path.dirname(class_text_path), exist_ok=True)
            class_texts = {}
            for class_name in self.class_names:
                class_texts[class_name] = [class_name, f"a {class_name}", f"the {class_name}"]
            
            with open(class_text_path, 'w') as f:
                json.dump(class_texts, f, indent=2)
            
            print(f"✓ Created sample class text file at: {class_text_path}")
        else:
            print(f"✓ Class text file found: {class_text_path}")
            
            # Load and display contents
            try:
                with open(class_text_path, 'r') as f:
                    class_texts = json.load(f)
                
                # Handle both list and dict formats
                if isinstance(class_texts, list):
                    print(f"  - Contains {len(class_texts)} class names (list format)")
                    print(f"  - Classes: {class_texts}")
                elif isinstance(class_texts, dict):
                    print(f"  - Contains {len(class_texts)} classes (dict format)")
                    print("  - Text prompts per class:")
                    for class_name, texts in class_texts.items():
                        print(f"    {class_name}: {texts}")
                else:
                    print(f"  - Unexpected format: {type(class_texts)}")
                    
            except Exception as e:
                print(f"✗ Error reading class text file: {e}")


def main():
    """Main function to run the dataloader visualization."""
    print("YOLO-World Dataloader Visualization")
    print("=" * 50)
    
    visualizer = DataloaderVisualizer()
    
    # Check class text file
    visualizer.check_class_text_file('data/texts/flir_v2_16_classes.json')
    
    # Visualize all splits
    visualizer.visualize_all_splits()
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print("\nGenerated files:")
    print("  - dataloader_visualizations/train_samples.png")
    print("  - dataloader_visualizations/validation_samples.png") 
    print("  - dataloader_visualizations/test_samples.png")
    print("  - dataloader_visualizations/dataset_summary.png")
    print("\nTo use with mmdetection (if installed), you can also:")
    print("1. Load your config: cfg = Config.fromfile('your_config.py')")
    print("2. Build datasets: train_dataset = build_dataset(cfg.train_dataloader.dataset)")
    print("3. Iterate through samples to verify data loading")


if __name__ == "__main__":
    main()