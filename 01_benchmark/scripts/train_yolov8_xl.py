#!/usr/bin/env python3
"""
Restart training for YOLOv8-XL on RTX 5060 with Windows-safe settings.
Fixed: workers=0 to prevent Windows dataloader deadlock.
"""

import torch
from ultralytics import YOLO
import os

def main():
    print("="*80)
    print(" YOLOv8-XL RESTART TRAINING (WINDOWS-SAFE)")
    print("="*80)
    
    # GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    
    print("\n" + "-"*80)
    print(" SAFE CONFIGURATION (WINDOWS FIXES)")
    print("-"*80)
    
    config = {
        'Model': 'YOLOv8-XL',
        'Image size': 1280,
        'Batch size': 1,
        'Epochs': 150,
        'Early stopping patience': 50,
        'AMP enabled': True,
        'Workers': 0,
        'Cache': False,
        'Optimizer': 'AdamW',
        'Learning rate': 0.01,
    }
    
    for key, value in config.items():
        print(f"  {key:25} {value}")
    
    print("\nWindows-specific fixes:")
    print("  ✓ workers=0: Prevents dataloader deadlock")
    print("  ✓ cache=False: Prevents RAM caching issues")
    print("  ✓ batch=1: Safe VRAM usage (~4-5GB)")
    
    print("\n" + "-"*80)
    print(" STARTING TRAINING")
    print("-"*80)
    print("Expected timeline:")
    print("  Epoch 1: ~40 minutes (check for progress within 3 minutes)")
    print("  Checkpoints: Every 10 epochs")
    print("  Total time: 75-100 hours (3-4 days)")
    print("  Target mAP@0.5: >0.56")
    print("-"*80)
    
    # Load model
    model = YOLO('yolov8x.pt')
    
    # Start training with Windows-safe settings
    results = model.train(
        # Dataset
        data='D:/projects/drone-detection-project/01_benchmark/data/visdrone_simple.yaml',
        
        # Training parameters
        epochs=150,
        patience=50,  # Early stopping
        imgsz=1280,
        batch=1,  # Safe for Windows + 8GB VRAM
        device='0',
        workers=0,  # CRITICAL: Windows dataloader deadlock fix
        cache=False,  # Prevent RAM caching issues
        
        # Optimization
        amp=True,  # Mixed precision for Blackwell
        cos_lr=True,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Regularization
        label_smoothing=0.1,
        dropout=0.1,
        
        # Augmentation (critical for small objects)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # Keep 0 for aerial
        translate=0.1,
        scale=0.5,
        flipud=0.0,  # Keep 0 for aerial
        fliplr=0.5,
        mosaic=1.0,  # Essential
        mixup=0.1,
        copy_paste=0.3,  # Critical for VisDrone
        
        # Output
        save=True,
        save_period=10,  # Checkpoint every 10 epochs
        project='runs/safe',
        name='yolov8_xl_restart',
        exist_ok=True,
        verbose=True,
        plots=True,  # Generate training plots
    )
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE")
    print("="*80)
    
    if hasattr(results, 'results_dict'):
        print("\nFinal metrics:")
        for key, value in results.results_dict.items():
            if 'metrics' in key.lower():
                print(f"  {key}: {value:.4f}")
    
    print(f"\nModel saved to: runs/safe/yolov8_xl_restart/weights/")
    print("Next: Train YOLOv11-X with same safe settings")
    
    return results

if __name__ == '__main__':
    main()