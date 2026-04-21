#!/usr/bin/env python3
"""
Convert VisDrone dataset annotations to YOLO format.

VisDrone annotation format (8 values per line):
1. bbox_left: Bounding box left coordinate
2. bbox_top: Bounding box top coordinate  
3. width: Bounding box width
4. height: Bounding box height
5. score: Detection score (1 for GT)
6. object_category: Object category (0-11)
7. truncation: Truncation level (0-2)
8. occlusion: Occlusion level (0-2)

YOLO format (normalized):
<class_id> <x_center> <y_center> <width> <height>

VisDrone categories (12 classes):
0: pedestrian
1: people
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor
10: others
11: ignore

We'll map to 10 classes (combine pedestrian/people, ignore 'others' and 'ignore')
"""

import os
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
import cv2

# VisDrone to YOLO class mapping
# We'll use 10 classes as per reference project
CLASS_MAPPING = {
    0: 0,  # pedestrian -> person
    1: 0,  # people -> person
    2: 1,  # bicycle
    3: 2,  # car
    4: 3,  # van
    5: 4,  # truck
    6: 5,  # tricycle
    7: 6,  # awning-tricycle
    8: 7,  # bus
    9: 8,  # motor
    # 10: others (ignore)
    # 11: ignore (ignore)
}

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

def convert_annotation(ann_path, img_width, img_height):
    """Convert VisDrone annotation to YOLO format."""
    yolo_annotations = []
    
    with open(ann_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) < 8:
            continue
            
        bbox_left = float(parts[0])
        bbox_top = float(parts[1])
        bbox_width = float(parts[2])
        bbox_height = float(parts[3])
        object_category = int(parts[5])
        
        # Skip ignored categories (others=10, ignore=11)
        if object_category >= 10:
            continue
            
        # Map to YOLO class
        if object_category not in CLASS_MAPPING:
            continue
            
        yolo_class = CLASS_MAPPING[object_category]
        
        # Calculate normalized coordinates
        x_center = (bbox_left + bbox_width / 2) / img_width
        y_center = (bbox_top + bbox_height / 2) / img_height
        width_norm = bbox_width / img_width
        height_norm = bbox_height / img_height
        
        # Ensure coordinates are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        # Skip if bbox is too small (optional)
        if width_norm < 0.001 or height_norm < 0.001:
            continue
            
        yolo_annotations.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
    
    return yolo_annotations

def process_dataset(split_name, raw_dir, yolo_dir, copy_images=True):
    """Process a dataset split (train/val/test)."""
    print(f"\nProcessing {split_name} set...")
    
    # Create directories
    images_dir = yolo_dir / split_name / 'images'
    labels_dir = yolo_dir / split_name / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get annotation files
    ann_dir = raw_dir / split_name / 'annotations'
    img_dir = raw_dir / split_name / 'images'
    
    if not ann_dir.exists():
        print(f"Warning: Annotation directory not found: {ann_dir}")
        return
    
    annotation_files = list(ann_dir.glob('*.txt'))
    print(f"Found {len(annotation_files)} annotation files")
    
    processed_count = 0
    skipped_count = 0
    
    for ann_file in tqdm(annotation_files, desc=f"Converting {split_name}"):
        # Get corresponding image
        img_file = img_dir / ann_file.name.replace('.txt', '.jpg')
        if not img_file.exists():
            # Try with .png extension
            img_file = img_dir / ann_file.name.replace('.txt', '.png')
            if not img_file.exists():
                skipped_count += 1
                continue
        
        # Read image dimensions
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                skipped_count += 1
                continue
            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"Error reading {img_file}: {e}")
            skipped_count += 1
            continue
        
        # Convert annotations
        yolo_anns = convert_annotation(ann_file, img_width, img_height)
        
        # Skip if no valid annotations
        if not yolo_anns:
            skipped_count += 1
            continue
        
        # Save YOLO annotations
        label_file = labels_dir / ann_file.name
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_anns))
        
        # Copy image
        if copy_images:
            dest_img = images_dir / img_file.name
            shutil.copy2(img_file, dest_img)
        
        processed_count += 1
    
    print(f"Processed: {processed_count}, Skipped: {skipped_count}")
    return processed_count

def create_dataset_yaml(yolo_dir, output_path):
    """Create YOLO dataset YAML file."""
    yaml_content = f"""# VisDrone Dataset YAML
# Classes: {len(CLASS_NAMES)}

path: {yolo_dir.parent.absolute()}  # dataset root dir
train: visdrone_yolo/train/images  # train images
val: visdrone_yolo/val/images      # val images
test: visdrone_yolo/test/images    # test images (optional)

# Classes
nc: {len(CLASS_NAMES)}  # number of classes
names: {CLASS_NAMES}

# Download command/URL (optional)
# download: https://github.com/VisDrone/VisDrone-Dataset

# License
# license: CC BY-NC-SA 4.0
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset YAML: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert VisDrone dataset to YOLO format')
    parser.add_argument('--raw-dir', type=str, default='data/visdrone_raw',
                       help='Path to raw VisDrone dataset')
    parser.add_argument('--yolo-dir', type=str, default='data/visdrone_yolo',
                       help='Path to output YOLO dataset')
    parser.add_argument('--no-copy-images', action='store_true',
                       help='Do not copy images (create symlinks instead)')
    parser.add_argument('--create-yaml', action='store_true',
                       help='Create dataset YAML file')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    raw_dir = Path(args.raw_dir)
    yolo_dir = Path(args.yolo_dir)
    
    # Check if raw directory exists
    if not raw_dir.exists():
        print(f"Error: Raw directory not found: {raw_dir}")
        print("Please make sure you have downloaded and extracted the VisDrone dataset.")
        return
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    total_processed = 0
    for split in splits:
        split_raw_dir = raw_dir / split
        
        if split_raw_dir.exists():
            processed = process_dataset(
                split, 
                raw_dir, 
                yolo_dir, 
                copy_images=not args.no_copy_images
            )
            if processed:
                total_processed += processed
        else:
            print(f"Warning: {split} directory not found: {split_raw_dir}")
    
    print(f"\nTotal processed files: {total_processed}")
    
    # Create dataset YAML
    if args.create_yaml:
        yaml_path = yolo_dir.parent / 'visdrone.yaml'
        create_dataset_yaml(yolo_dir, yaml_path)
        
        # Also create a simplified version for Ultralytics
        simple_yaml_path = yolo_dir.parent / 'visdrone_simple.yaml'
        with open(simple_yaml_path, 'w') as f:
            f.write(f"""# VisDrone Dataset for Ultralytics YOLO
path: {yolo_dir.parent.absolute()}
train: visdrone_yolo/train
val: visdrone_yolo/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")
        print(f"Created simplified YAML: {simple_yaml_path}")
    
    print("\nConversion complete!")
    print(f"YOLO dataset saved to: {yolo_dir}")
    print(f"Class mapping: {CLASS_NAMES}")

if __name__ == '__main__':
    main()