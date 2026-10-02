#!/usr/bin/env python3
"""
SAHI evaluation for small object detection.
Uses correct API: get_prediction() and model_type='ultralytics'.
"""

import os
import json
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image
from ultralytics import YOLO

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "01_benchmark" / "data" / "visdrone_simple.yaml"
VAL_IMAGES = REPO_ROOT / "01_benchmark" / "data" / "visdrone_yolo" / "val" / "images"


def evaluate_sahi(model_path, test_images_dir, output_file='sahi_results.json'):
    """
    Compare SAHI vs standard inference.
    
    Args:
        model_path: Path to trained model (.pt file)
        test_images_dir: Directory with test images
        output_file: JSON file to save results
    """
    
    print("="*80)
    print(" SAHI EVALUATION FOR SMALL OBJECT DETECTION")
    print("="*80)
    
    # Universal model type for all Ultralytics models
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',  # Works for v8/v10/v11
        model_path=model_path,
        confidence_threshold=0.25,
        device='cuda:0'
    )
    
    # Get test images
    test_images = []
    if os.path.exists(test_images_dir):
        test_images = [os.path.join(test_images_dir, f) 
                      for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return None
    
    # Use first 10 images for evaluation
    test_images = test_images[:10]
    print(f"Testing on {len(test_images)} images...")
    
    results = []
    for i, img_path in enumerate(test_images, 1):
        try:
            image = read_image(img_path)
            
            # Standard inference (FIXED API)
            standard_result = get_prediction(image, detection_model)
            
            # SAHI sliced inference (640x640 windows)
            sahi_result = get_sliced_prediction(
                image,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            
            standard_count = len(standard_result.object_prediction_list)
            sahi_count = len(sahi_result.object_prediction_list)
            improvement = sahi_count - standard_count
            improvement_pct = (improvement / max(1, standard_count)) * 100
            
            results.append({
                'image': os.path.basename(img_path),
                'standard_detections': standard_count,
                'sahi_detections': sahi_count,
                'improvement': improvement,
                'improvement_percent': round(improvement_pct, 1)
            })
            
            print(f"  Image {i}: {standard_count} → {sahi_count} detections "
                  f"(+{improvement}, +{improvement_pct:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(img_path)}: {str(e)[:80]}")
            continue
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate summary
    if results:
        total_standard = sum(r['standard_detections'] for r in results)
        total_sahi = sum(r['sahi_detections'] for r in results)
        avg_improvement = sum(r['improvement_percent'] for r in results) / len(results)
        
        print("\n" + "-"*80)
        print(" SUMMARY")
        print("-"*80)
        print(f"Total standard detections: {total_standard}")
        print(f"Total SAHI detections: {total_sahi}")
        print(f"Average improvement: {avg_improvement:.1f}%")
        
        if avg_improvement > 3.0:
            print(f"✅ SAHI provides meaningful improvement (>3%)")
        else:
            print(f"⚠ SAHI improvement minimal (≤3%)")
        
        print(f"\nResults saved to: {output_file}")
    
    return results

def main():
    """Main evaluation function."""
    
    # Test directory (using validation images)
    test_dir = str(VAL_IMAGES)
    
    # Check which models are available
    model_dirs = [
        ('runs/final/yolov8_xl_final/weights/best.pt', 'YOLOv8-XL'),
        ('runs/final/yolo11x_final/weights/best.pt', 'YOLOv11-X'),
        ('runs/final/yolov10_x_final/weights/best.pt', 'YOLOv10-X'),
    ]
    
    available_models = []
    for model_path, model_name in model_dirs:
        if os.path.exists(model_path):
            available_models.append((model_path, model_name))
    
    if not available_models:
        print("❌ No trained models found. Train models first.")
        print("Run: python train_yolov8_xl_final.py")
        return
    
    print(f"Found {len(available_models)} trained model(s)")
    
    # Evaluate best model first (your recommendation)
    best_model_path, best_model_name = available_models[0]
    print(f"\nEvaluating {best_model_name} first...")
    
    results = evaluate_sahi(best_model_path, test_dir, 
                           f'sahi_{best_model_name.lower()}.json')
    
    if results:
        avg_improvement = sum(r['improvement_percent'] for r in results) / len(results)
        
        print("\n" + "="*80)
        print(" RECOMMENDATION")
        print("="*80)
        
        if avg_improvement > 3.0:
            print(f"SAHI improves detection by {avg_improvement:.1f}%")
            print("✅ Worth applying to all models")
            
            # Evaluate remaining models
            for model_path, model_name in available_models[1:]:
                print(f"\nEvaluating {model_name}...")
                evaluate_sahi(model_path, test_dir, f'sahi_{model_name.lower()}.json')
        else:
            print(f"SAHI improvement only {avg_improvement:.1f}%")
            print("⚠ Not worth applying to other models")
    
    print("\n" + "="*80)
    print(" EVALUATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()