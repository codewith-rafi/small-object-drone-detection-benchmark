#!/usr/bin/env python3
"""
P2 Head Modification for YOLOv8-XL.
Adds high-resolution P2 detection head (1/4 scale) for small object detection.
Based on YOLOv8 architecture analysis and literature review.
"""

import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path

def create_p2_yolov8x_config():
    """
    Create YOLOv8-XL configuration with P2 head.
    Modifies the YAML configuration to add P2 detection head.
    """
    
    # Original YOLOv8-XL configuration (from Ultralytics)
    original_config = {
        'nc': 10,  # Number of classes (VisDrone)
        'scales': {
            'x': {
                'depth_multiple': 1.00,
                'width_multiple': 1.25,
                'backbone': [
                    [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
                    [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
                    [-1, 3, 'C2f', [128, True]],
                    [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
                    [-1, 6, 'C2f', [256, True]],
                    [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
                    [-1, 6, 'C2f', [512, True]],
                    [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
                    [-1, 3, 'C2f', [1024, True]],
                    [-1, 1, 'SPPF', [1024, 5]],  # 9
                ],
                'head': [
                    [-1, 1, 'nn.Upsample', ['nearest']],
                    [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                    [-1, 3, 'C2f', [512]],  # 12
                    
                    [-1, 1, 'nn.Upsample', ['nearest']],
                    [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                    [-1, 3, 'C2f', [256]],  # 15 (P3/8)
                    
                    [-1, 1, 'Conv', [256, 3, 2]],
                    [[-1, 12], 1, 'Concat', [1]],  # cat head P4
                    [-1, 3, 'C2f', [512]],  # 18 (P4/16)
                    
                    [-1, 1, 'Conv', [512, 3, 2]],
                    [[-1, 9], 1, 'Concat', [1]],  # cat backbone P5
                    [-1, 3, 'C2f', [1024]],  # 21 (P5/32)
                    
                    # Original detection heads (P3, P4, P5)
                    [[15, 18, 21], 1, 'Detect', ['nc']],  # Detect(P3, P4, P5)
                ]
            }
        }
    }
    
    # Modified configuration with P2 head
    p2_config = {
        'nc': 10,  # Number of classes (VisDrone)
        'scales': {
            'x': {
                'depth_multiple': 1.00,
                'width_multiple': 1.25,
                'backbone': [
                    [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
                    [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
                    [-1, 3, 'C2f', [128, True]],   # 2-P2 feature
                    [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
                    [-1, 6, 'C2f', [256, True]],   # 4-P3 feature
                    [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
                    [-1, 6, 'C2f', [512, True]],   # 6-P4 feature
                    [-1, 1, 'Conv', [1024, 3, 2]], # 7-P5/32
                    [-1, 3, 'C2f', [1024, True]],  # 8-P5 feature
                    [-1, 1, 'SPPF', [1024, 5]],    # 9
                ],
                'head': [
                    # Path to P4
                    [-1, 1, 'nn.Upsample', ['nearest']],
                    [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                    [-1, 3, 'C2f', [512]],  # 12
                    
                    # Path to P3
                    [-1, 1, 'nn.Upsample', ['nearest']],
                    [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                    [-1, 3, 'C2f', [256]],  # 15 (P3/8)
                    
                    # NEW: Path to P2 (high resolution for small objects)
                    [-1, 1, 'nn.Upsample', ['nearest']],
                    [[-1, 2], 1, 'Concat', [1]],  # cat backbone P2
                    [-1, 3, 'C2f', [128]],  # 18 (P2/4) - NEW P2 HEAD
                    
                    # Downsample paths
                    [-1, 1, 'Conv', [128, 3, 2]],  # P2 -> P3
                    [[-1, 15], 1, 'Concat', [1]],  # cat with P3
                    [-1, 3, 'C2f', [256]],  # 21
                    
                    [-1, 1, 'Conv', [256, 3, 2]],  # P3 -> P4
                    [[-1, 12], 1, 'Concat', [1]],  # cat with P4
                    [-1, 3, 'C2f', [512]],  # 24
                    
                    [-1, 1, 'Conv', [512, 3, 2]],  # P4 -> P5
                    [[-1, 9], 1, 'Concat', [1]],  # cat with P5
                    [-1, 3, 'C2f', [1024]],  # 27
                    
                    # Detection heads: P2, P3, P4, P5 (4 heads total)
                    [[18, 21, 24, 27], 1, 'Detect', ['nc']],  # Detect(P2, P3, P4, P5)
                ]
            }
        }
    }
    
    return p2_config

def save_yaml_config(config, output_path):
    """Save YAML configuration to file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Configuration saved to: {output_path}")
    return output_path

def create_p2_model_from_checkpoint(checkpoint_path, output_model_path):
    """
    Create P2-modified model from existing checkpoint.
    This is a placeholder - actual implementation requires modifying the model architecture.
    """
    print(f"⚠️  Note: Full P2 model creation from checkpoint requires custom implementation.")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output: {output_model_path}")
    print(f"  This would involve:")
    print(f"  1. Loading original model weights")
    print(f"  2. Creating new model with P2 architecture")
    print(f"  3. Transferring compatible weights")
    print(f"  4. Initializing new P2 head weights")
    
    # For now, create the configuration file
    config = create_p2_yolov8x_config()
    config_path = output_model_path.replace('.pt', '.yaml')
    save_yaml_config(config, config_path)
    
    return config_path

def train_p2_model(config_path, data_yaml, output_dir='runs/p2_yolov8_xl'):
    """
    Training script for P2-modified YOLOv8-XL.
    """
    import subprocess
    import sys
    
    cmd = [
        sys.executable, '-m', 'ultralytics', 'train',
        'model', config_path,
        'data', data_yaml,
        'imgsz', '1280',
        'batch', '1',  # Conservative for P2 head (higher resolution)
        'epochs', '150',
        'patience', '50',
        'save_period', '20',
        'workers', '0',  # Windows-safe
        'amp', 'True',
        'seed', '0',
        'deterministic', 'true',
        'project', output_dir,
        'name', 'p2_yolov8_xl',
        'exist_ok', 'True'
    ]
    
    print(f"\nTraining command:")
    print(' '.join(cmd))
    
    # Note: This would be executed after baseline training completes
    print(f"\n⚠️  Execute this after baseline training completes.")
    print(f"  The P2 model will be trained from scratch with the modified architecture.")
    
    return cmd

def compare_architectures():
    """Compare original vs P2-modified architectures."""
    
    original = create_p2_yolov8x_config()  # Actually returns P2 config
    print("="*80)
    print(" ARCHITECTURE COMPARISON")
    print("="*80)
    
    print("\nORIGINAL YOLOv8-XL:")
    print("  - Detection heads: P3, P4, P5 (3 heads)")
    print("  - Smallest feature map: 1/8 scale (P3)")
    print("  - Best for: Medium to large objects")
    
    print("\nP2-MODIFIED YOLOv8-XL:")
    print("  - Detection heads: P2, P3, P4, P5 (4 heads)")
    print("  - Smallest feature map: 1/4 scale (P2)")
    print("  - Best for: Small objects (critical for VisDrone)")
    print("  - Higher resolution: 4× more spatial information than P3")
    print("  - Memory impact: ~15-20% increase due to P2 head")
    print("  - Expected benefit: +5-10% mAP for small objects")
    
    print("\n" + "-"*80)
    print(" RESEARCH JUSTIFICATION")
    print("-"*80)
    print("VisDrone dataset: 66.2% objects < 32px")
    print("P2 head provides 1/4 scale features vs 1/8 scale in original")
    print("This matches object size distribution better")
    print("Expected to improve small object detection significantly")

def main():
    """Main function for P2 head modification."""
    
    print("="*80)
    print(" P2 HEAD MODIFICATION FOR YOLOv8-XL")
    print("="*80)
    print("Adding high-resolution detection head for small object detection")
    print("Target: VisDrone dataset (66.2% objects < 32px)")
    print("="*80)
    
    # Compare architectures
    compare_architectures()
    
    # Create configuration
    config = create_p2_yolov8x_config()
    
    # Save configuration
    output_dir = 'p2_configs'
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'yolov8x_p2.yaml')
    save_yaml_config(config, config_path)
    
    # Show training command
    print("\n" + "-"*80)
    print(" TRAINING PREPARATION")
    print("-"*80)
    
    from pathlib import Path
    data_yaml = str(Path(__file__).resolve().parents[2] / '01_benchmark/data/visdrone_simple.yaml')
    train_cmd = train_p2_model(config_path, data_yaml)
    
    print("\n" + "="*80)
    print(" NEXT STEPS")
    print("="*80)
    print("1. Complete baseline YOLOv8-XL training")
    print("2. Run SAHI parameter study")
    print("3. Train P2-modified model using above configuration")
    print("4. Conduct interaction study (Baseline vs +SAHI vs +P2 vs +Both)")
    print("5. Calculate interaction metrics and analyze results")
    
    print(f"\n✅ P2 configuration ready at: {config_path}")
    print("⚠️  Note: P2 training requires ~15-20% more VRAM")
    print("   Adjust batch size if needed (start with batch=1)")

if __name__ == '__main__':
    main()