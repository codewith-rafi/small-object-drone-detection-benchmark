#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for VisDrone dataset.
Analyzes class distribution, image dimensions, bounding box sizes, etc.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import yaml

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Class names (9 classes as per our conversion)
CLASS_NAMES = [
    'person',      # 0
    'bicycle',     # 1
    'car',         # 2
    'van',         # 3
    'truck',       # 4
    'tricycle',    # 5
    'awning-tricycle',  # 6
    'bus',         # 7
    'motor',       # 8
]

def read_yolo_annotation(ann_path, img_width, img_height):
    """Read YOLO format annotation and convert to pixel coordinates."""
    bboxes = []
    class_ids = []
    
    with open(ann_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 5:
            continue
            
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Convert to pixel coordinates
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        x2 = (x_center + width/2) * img_width
        y2 = (y_center + height/2) * img_height
        
        bboxes.append([x1, y1, x2, y2])
        class_ids.append(class_id)
    
    return np.array(bboxes), np.array(class_ids)

def analyze_dataset_split(split_name, data_dir):
    """Analyze a dataset split (train/val/test)."""
    print(f"\nAnalyzing {split_name} set...")
    
    images_dir = data_dir / split_name / 'images'
    labels_dir = data_dir / split_name / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Warning: {split_name} directory not found or incomplete")
        return None
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"Found {len(image_files)} images")
    
    # Initialize statistics
    stats = {
        'image_files': [],
        'image_heights': [],
        'image_widths': [],
        'num_objects': [],
        'class_counts': np.zeros(len(CLASS_NAMES), dtype=int),
        'bbox_widths': [],
        'bbox_heights': [],
        'bbox_areas': [],
        'aspect_ratios': [],
        'small_objects': 0,  # objects < 32x32 pixels
        'medium_objects': 0, # objects 32x32 to 96x96
        'large_objects': 0,  # objects > 96x96
    }
    
    # Process each image
    for img_file in tqdm(image_files[:1000], desc=f"Processing {split_name}"):  # Limit to 1000 for speed
        # Get corresponding label file
        label_file = labels_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            continue
        
        # Read image dimensions
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"Error reading {img_file}: {e}")
            continue
        
        # Read annotations
        bboxes, class_ids = read_yolo_annotation(label_file, img_width, img_height)
        
        # Update statistics
        stats['image_files'].append(img_file.name)
        stats['image_heights'].append(img_height)
        stats['image_widths'].append(img_width)
        stats['num_objects'].append(len(bboxes))
        
        # Update class counts
        for class_id in class_ids:
            if 0 <= class_id < len(CLASS_NAMES):
                stats['class_counts'][class_id] += 1
        
        # Analyze bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            stats['bbox_widths'].append(width)
            stats['bbox_heights'].append(height)
            stats['bbox_areas'].append(area)
            stats['aspect_ratios'].append(aspect_ratio)
            
            # Categorize by size (COCO standard)
            if area < 32*32:
                stats['small_objects'] += 1
            elif area <= 96*96:
                stats['medium_objects'] += 1
            else:
                stats['large_objects'] += 1
    
    return stats

def plot_statistics(stats_dict, output_dir):
    """Create visualization plots for dataset statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Class Distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('VisDrone Dataset Statistics', fontsize=16)
    
    # Class distribution (bar chart)
    ax = axes[0, 0]
    splits = list(stats_dict.keys())
    class_data = []
    
    for split in splits:
        if stats_dict[split] is not None:
            class_data.append(stats_dict[split]['class_counts'])
    
    if class_data:
        class_data = np.array(class_data)
        x = np.arange(len(CLASS_NAMES))
        width = 0.8 / len(splits)
        
        for i, split in enumerate(splits):
            if stats_dict[split] is not None:
                ax.bar(x + i*width - width*(len(splits)-1)/2, 
                      class_data[i], width, label=split)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Image Dimensions
    ax = axes[0, 1]
    all_heights = []
    all_widths = []
    split_labels = []
    
    for split in splits:
        if stats_dict[split] is not None:
            all_heights.extend(stats_dict[split]['image_heights'])
            all_widths.extend(stats_dict[split]['image_widths'])
            split_labels.extend([split] * len(stats_dict[split]['image_heights']))
    
    if all_heights and all_widths:
        df = pd.DataFrame({
            'Height': all_heights,
            'Width': all_widths,
            'Split': split_labels
        })
        
        sns.scatterplot(data=df, x='Width', y='Height', hue='Split', alpha=0.6, ax=ax)
        ax.set_title('Image Dimensions')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        ax.grid(True, alpha=0.3)
    
    # 3. Objects per Image
    ax = axes[0, 2]
    all_counts = []
    split_labels = []
    
    for split in splits:
        if stats_dict[split] is not None:
            all_counts.extend(stats_dict[split]['num_objects'])
            split_labels.extend([split] * len(stats_dict[split]['num_objects']))
    
    if all_counts:
        df = pd.DataFrame({
            'Objects per Image': all_counts,
            'Split': split_labels
        })
        
        sns.boxplot(data=df, x='Split', y='Objects per Image', ax=ax)
        ax.set_title('Objects per Image Distribution')
        ax.grid(True, alpha=0.3)
    
    # 4. Bounding Box Sizes
    ax = axes[1, 0]
    all_widths = []
    all_heights = []
    split_labels = []
    
    for split in splits:
        if stats_dict[split] is not None:
            all_widths.extend(stats_dict[split]['bbox_widths'])
            all_heights.extend(stats_dict[split]['bbox_heights'])
            split_labels.extend([split] * len(stats_dict[split]['bbox_widths']))
    
    if all_widths and all_heights:
        # Log scale for better visualization
        df = pd.DataFrame({
            'Width': all_widths,
            'Height': all_heights,
            'Split': split_labels
        })
        
        # Sample for scatter plot (too many points)
        df_sample = df.sample(min(10000, len(df)), random_state=42)
        sns.scatterplot(data=df_sample, x='Width', y='Height', hue='Split', alpha=0.3, ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Bounding Box Sizes (log scale)')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        ax.grid(True, alpha=0.3)
    
    # 5. Object Size Distribution
    ax = axes[1, 1]
    size_data = []
    split_labels = []
    size_types = []
    
    for split in splits:
        if stats_dict[split] is not None:
            stats = stats_dict[split]
            total = stats['small_objects'] + stats['medium_objects'] + stats['large_objects']
            if total > 0:
                size_data.extend([
                    stats['small_objects'] / total * 100,
                    stats['medium_objects'] / total * 100,
                    stats['large_objects'] / total * 100
                ])
                split_labels.extend([split] * 3)
                size_types.extend(['Small (<32px)', 'Medium (32-96px)', 'Large (>96px)'])
    
    if size_data:
        df = pd.DataFrame({
            'Percentage': size_data,
            'Split': split_labels,
            'Size Category': size_types
        })
        
        sns.barplot(data=df, x='Split', y='Percentage', hue='Size Category', ax=ax)
        ax.set_title('Object Size Distribution (%)')
        ax.set_ylabel('Percentage')
        ax.grid(True, alpha=0.3)
    
    # 6. Aspect Ratios
    ax = axes[1, 2]
    all_ratios = []
    split_labels = []
    
    for split in splits:
        if stats_dict[split] is not None:
            ratios = stats_dict[split]['aspect_ratios']
            # Filter extreme aspect ratios for better visualization
            ratios_filtered = [r for r in ratios if 0.1 <= r <= 10]
            all_ratios.extend(ratios_filtered)
            split_labels.extend([split] * len(ratios_filtered))
    
    if all_ratios:
        df = pd.DataFrame({
            'Aspect Ratio': all_ratios,
            'Split': split_labels
        })
        
        sns.histplot(data=df, x='Aspect Ratio', hue='Split', bins=50, alpha=0.6, ax=ax)
        ax.set_title('Bounding Box Aspect Ratios')
        ax.set_xlabel('Width/Height Ratio')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5)  # Limit to reasonable range
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics table
    summary_data = []
    for split in splits:
        if stats_dict[split] is not None:
            stats = stats_dict[split]
            summary_data.append({
                'Split': split,
                'Images': len(stats['image_files']),
                'Total Objects': int(stats['class_counts'].sum()),
                'Avg Objects/Image': np.mean(stats['num_objects']) if stats['num_objects'] else 0,
                'Avg Image Width': np.mean(stats['image_widths']) if stats['image_widths'] else 0,
                'Avg Image Height': np.mean(stats['image_heights']) if stats['image_heights'] else 0,
                'Avg Bbox Width': np.mean(stats['bbox_widths']) if stats['bbox_widths'] else 0,
                'Avg Bbox Height': np.mean(stats['bbox_heights']) if stats['bbox_heights'] else 0,
                'Small Objects %': stats['small_objects'] / (stats['small_objects'] + stats['medium_objects'] + stats['large_objects']) * 100 if (stats['small_objects'] + stats['medium_objects'] + stats['large_objects']) > 0 else 0,
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_dir / 'dataset_summary.csv', index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(df_summary.to_string(index=False))
        
        # Save as markdown for report
        with open(output_dir / 'dataset_summary.md', 'w') as f:
            f.write("# VisDrone Dataset Summary\n\n")
            f.write(df_summary.to_markdown(index=False))
    
    return df_summary if summary_data else None

def main():
    """Main function to run EDA."""
    # Set paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'visdrone_yolo'
    output_dir = base_dir / 'eda_results'
    
    print("="*80)
    print("VISDRONE DATASET EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Analyze each split
    splits = ['train', 'val', 'test']
    stats_dict = {}
    
    for split in splits:
        stats = analyze_dataset_split(split, data_dir)
        stats_dict[split] = stats
        
        if stats is not None:
            print(f"\n{split.upper()} SET STATISTICS:")
            print(f"  Images: {len(stats['image_files'])}")
            print(f"  Total objects: {int(stats['class_counts'].sum())}")
            print(f"  Average objects per image: {np.mean(stats['num_objects']):.2f}")
            print(f"  Image dimensions: {np.mean(stats['image_widths']):.0f}x{np.mean(stats['image_heights']):.0f}")
            print(f"  Class distribution:")
            for i, class_name in enumerate(CLASS_NAMES):
                count = stats['class_counts'][i]
                if count > 0:
                    print(f"    {class_name}: {count}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS...")
    print("="*80)
    
    summary_df = plot_statistics(stats_dict, output_dir)
    
    # Save raw statistics
    with open(output_dir / 'dataset_statistics.json', 'w') as f:
        json_stats = {}
        for split in splits:
            if stats_dict[split] is not None:
                # Convert numpy arrays to lists for JSON serialization
                json_stats[split] = {
                    'image_count': len(stats_dict[split]['image_files']),
                    'total_objects': int(stats_dict[split]['class_counts'].sum()),
                    'class_counts': stats_dict[split]['class_counts'].tolist(),
                    'avg_image_width': float(np.mean(stats_dict[split]['image_widths'])) if stats_dict[split]['image_widths'] else 0,
                    'avg_image_height': float(np.mean(stats_dict[split]['image_heights'])) if stats_dict[split]['image_heights'] else 0,
                    'avg_objects_per_image': float(np.mean(stats_dict[split]['num_objects'])) if stats_dict[split]['num_objects'] else 0,
                    'small_objects': stats_dict[split]['small_objects'],
                    'medium_objects': stats_dict[split]['medium_objects'],
                    'large_objects': stats_dict[split]['large_objects'],
                }
        json.dump(json_stats, f, indent=2)
    
    print("\n" + "="*80)
    print("EDA COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*80)
    
    # Key findings
    print("\nKEY FINDINGS:")
    if summary_df is not None:
        train_stats = summary_df[summary_df['Split'] == 'train'].iloc[0]
        print(f"1. Small objects: {train_stats['Small Objects %']:.1f}% of objects are <32px")
        print(f"2. Average bbox size: {train_stats['Avg Bbox Width']:.1f}x{train_stats['Avg Bbox Height']:.1f} pixels")
        print(f"3. Objects per image: {train_stats['Avg Objects/Image']:.1f} on average")
        print(f"4. Image resolution: {train_stats['Avg Image Width']:.0f}x{train_stats['Avg Image Height']:.0f}")
    
    print("\nRECOMMENDATIONS FOR TRAINING:")
    print("1. Use SAHI (Slicing Aided Hyper Inference) for small object detection")
    print("2. Implement mosaic augmentation with small object focus")
    print("3. Consider using higher input resolution (e.g., 1280x1280)")
    print("4. Use class-balanced sampling if class distribution is uneven")
    print("5. Monitor small object recall during training")

if __name__ == '__main__':
    main()