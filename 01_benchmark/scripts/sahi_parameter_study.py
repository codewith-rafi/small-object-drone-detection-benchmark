#!/usr/bin/env python3
"""
SAHI parameter sensitivity study for VisDrone dataset.
4x4 grid: slice sizes (320, 640, 1024, 1280) x overlap ratios (0.1, 0.2, 0.3, 0.4)
Output: mAP vs FPS trade-off curves for paper.
"""

import os
import json
import time
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def run_sahi_parameter_study(model_path, test_images_dir, output_dir='sahi_results'):
    """
    Run comprehensive SAHI parameter study.
    
    Args:
        model_path: Path to trained model (.pt file)
        test_images_dir: Directory with test images
        output_dir: Directory to save results
    """
    
    print("="*80)
    print(" SAHI PARAMETER SENSITIVITY STUDY")
    print("="*80)
    print("Parameter grid: 4 slice sizes × 4 overlap ratios = 16 configurations")
    print("Slice sizes: 320, 640, 1024, 1280")
    print("Overlap ratios: 0.1, 0.2, 0.3, 0.4")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model once (reused for all configurations)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=0.25,
        device='cuda:0'
    )
    
    # Get test images (use validation set for consistency)
    test_images = []
    if os.path.exists(test_images_dir):
        test_images = [os.path.join(test_images_dir, f) 
                      for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return None
    
    # Use subset for parameter study (20 images for speed)
    test_images = test_images[:20]
    print(f"Testing on {len(test_images)} images...")
    
    # Parameter grid
    slice_sizes = [320, 640, 1024, 1280]
    overlap_ratios = [0.1, 0.2, 0.3, 0.4]
    
    results = []
    
    # First get baseline (no SAHI) for comparison
    print("\n" + "-"*80)
    print(" BASELINE (No SAHI)")
    print("-"*80)
    
    baseline_results = []
    baseline_start = time.time()
    
    for i, img_path in enumerate(test_images, 1):
        try:
            image = read_image(img_path)
            result = get_prediction(image, detection_model)
            
            detections = len(result.object_prediction_list)
            baseline_results.append({
                'image': os.path.basename(img_path),
                'detections': detections
            })
            
            if i % 5 == 0:
                print(f"  Processed {i}/{len(test_images)} images")
                
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(img_path)}: {str(e)[:80]}")
            continue
    
    baseline_time = time.time() - baseline_start
    avg_baseline_detections = np.mean([r['detections'] for r in baseline_results])
    baseline_fps = len(test_images) / baseline_time
    
    print(f"  Baseline: {avg_baseline_detections:.1f} avg detections, {baseline_fps:.1f} FPS")
    
    # Now test all SAHI configurations
    print("\n" + "-"*80)
    print(" SAHI PARAMETER SWEEP")
    print("-"*80)
    
    for slice_size in slice_sizes:
        for overlap in overlap_ratios:
            print(f"\nTesting: slice={slice_size}, overlap={overlap}")
            
            config_results = []
            config_start = time.time()
            
            for i, img_path in enumerate(test_images, 1):
                try:
                    image = read_image(img_path)
                    
                    # SAHI sliced inference
                    sahi_result = get_sliced_prediction(
                        image,
                        detection_model,
                        slice_height=slice_size,
                        slice_width=slice_size,
                        overlap_height_ratio=overlap,
                        overlap_width_ratio=overlap
                    )
                    
                    detections = len(sahi_result.object_prediction_list)
                    config_results.append({
                        'image': os.path.basename(img_path),
                        'detections': detections
                    })
                    
                except Exception as e:
                    print(f"  ❌ Error: {str(e)[:80]}")
                    continue
            
            config_time = time.time() - config_start
            avg_detections = np.mean([r['detections'] for r in config_results])
            config_fps = len(test_images) / config_time if config_time > 0 else 0
            
            # Calculate improvement over baseline
            improvement = avg_detections - avg_baseline_detections
            improvement_pct = (improvement / max(1, avg_baseline_detections)) * 100
            
            # Store results
            result_entry = {
                'slice_size': slice_size,
                'overlap_ratio': overlap,
                'avg_detections': round(avg_detections, 2),
                'fps': round(config_fps, 2),
                'improvement': round(improvement, 2),
                'improvement_pct': round(improvement_pct, 2),
                'time_per_image': round(config_time / len(test_images), 3),
                'baseline_detections': round(avg_baseline_detections, 2),
                'baseline_fps': round(baseline_fps, 2)
            }
            
            results.append(result_entry)
            
            print(f"  Results: {avg_detections:.1f} detections (+{improvement_pct:.1f}%), {config_fps:.1f} FPS")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'sahi_parameter_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, 'sahi_parameter_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'baseline': {
                'avg_detections': avg_baseline_detections,
                'fps': baseline_fps
            },
            'configurations': results,
            'test_images_count': len(test_images)
        }, f, indent=2)
    
    # Generate visualization
    generate_visualizations(df, output_dir)
    
    # Find optimal configuration
    find_optimal_configuration(df, output_dir)
    
    return df

def generate_visualizations(df, output_dir):
    """Generate visualization plots for paper."""
    
    print("\n" + "-"*80)
    print(" GENERATING VISUALIZATIONS")
    print("-"*80)
    
    # Plot 1: mAP (detections) vs FPS trade-off
    plt.figure(figsize=(10, 6))
    
    # Color by slice size
    colors = {320: 'blue', 640: 'green', 1024: 'orange', 1280: 'red'}
    
    for slice_size in df['slice_size'].unique():
        slice_data = df[df['slice_size'] == slice_size]
        plt.scatter(slice_data['fps'], slice_data['avg_detections'], 
                   c=colors[slice_size], s=100, alpha=0.7, 
                   label=f'Slice={slice_size}')
        
        # Add text labels for overlap ratios
        for _, row in slice_data.iterrows():
            plt.annotate(f"{row['overlap_ratio']}", 
                        (row['fps'], row['avg_detections']),
                        fontsize=8, alpha=0.7)
    
    plt.xlabel('FPS (Higher = Faster)')
    plt.ylabel('Average Detections (Higher = Better)')
    plt.title('SAHI Parameter Trade-off: Detection vs Speed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    tradeoff_path = os.path.join(output_dir, 'sahi_tradeoff_curve.png')
    plt.savefig(tradeoff_path, dpi=300, bbox_inches='tight')
    print(f"✅ Trade-off curve saved to: {tradeoff_path}")
    
    # Plot 2: Improvement heatmap
    plt.figure(figsize=(10, 8))
    
    # Create pivot table for heatmap
    pivot = df.pivot(index='slice_size', columns='overlap_ratio', values='improvement_pct')
    
    plt.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Improvement % over baseline')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.values[i, j]:.1f}%", 
                    ha='center', va='center', color='black', fontsize=10)
    
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('Overlap Ratio')
    plt.ylabel('Slice Size')
    plt.title('SAHI Parameter Heatmap: Improvement % over Baseline')
    
    heatmap_path = os.path.join(output_dir, 'sahi_parameter_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ Parameter heatmap saved to: {heatmap_path}")
    
    plt.close('all')

def find_optimal_configuration(df, output_dir):
    """Find optimal SAHI configuration based on different criteria."""
    
    print("\n" + "-"*80)
    print(" OPTIMAL CONFIGURATION ANALYSIS")
    print("-"*80)
    
    # Criteria 1: Best detection performance
    best_detection = df.loc[df['avg_detections'].idxmax()]
    print(f"🔍 Best detection: slice={best_detection['slice_size']}, "
          f"overlap={best_detection['overlap_ratio']}")
    print(f"   Detections: {best_detection['avg_detections']:.1f} "
          f"(+{best_detection['improvement_pct']:.1f}%), "
          f"FPS: {best_detection['fps']:.1f}")
    
    # Criteria 2: Best FPS (fastest)
    best_fps = df.loc[df['fps'].idxmax()]
    print(f"🔍 Fastest: slice={best_fps['slice_size']}, "
          f"overlap={best_fps['overlap_ratio']}")
    print(f"   FPS: {best_fps['fps']:.1f}, "
          f"Detections: {best_fps['avg_detections']:.1f} "
          f"(+{best_fps['improvement_pct']:.1f}%)")
    
    # Criteria 3: Best trade-off (detections per FPS)
    df['detections_per_fps'] = df['avg_detections'] / df['fps']
    best_tradeoff = df.loc[df['detections_per_fps'].idxmax()]
    print(f"🔍 Best trade-off: slice={best_tradeoff['slice_size']}, "
          f"overlap={best_tradeoff['overlap_ratio']}")
    print(f"   Detections/FPS: {best_tradeoff['detections_per_fps']:.3f}, "
          f"Detections: {best_tradeoff['avg_detections']:.1f}, "
          f"FPS: {best_tradeoff['fps']:.1f}")
    
    # Save optimal configurations
    optimal_configs = {
        'best_detection': best_detection.to_dict(),
        'best_fps': best_fps.to_dict(),
        'best_tradeoff': best_tradeoff.to_dict()
    }
    
    optimal_path = os.path.join(output_dir, 'optimal_configurations.json')
    with open(optimal_path, 'w') as f:
        json.dump(optimal_configs, f, indent=2)
    
    print(f"✅ Optimal configurations saved to: {optimal_path}")
    
    # Recommendation for paper
    print("\n" + "-"*80)
    print(" PAPER RECOMMENDATION")
    print("-"*80)
    print("For interaction study, consider using:")
    print(f"1. Best detection config for maximum accuracy")
    print(f"2. Best trade-off config for balanced performance")
    print(f"3. Include both in ablation study if space permits")

def main():
    """Main function for SAHI parameter study."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = Path(__file__).resolve().parent

    # Configuration — override locally after training
    model_path = str(scripts_dir / 'runs/detect/runs/safe/yolov8_xl_restart/weights/best.pt')
    test_images_dir = str(repo_root / '01_benchmark/data/visdrone_yolo/val/images')
    output_dir = str(repo_root / '01_benchmark/results/sahi')
    
    print("SAHI Parameter Study Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Test images: {test_images_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Parameter grid: 4×4 (16 configurations)")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model not found: {model_path}")
        print("Waiting for baseline training to complete...")
        print("This script should be run after baseline training finishes.")
        return
    
    # Run parameter study
    results = run_sahi_parameter_study(model_path, test_images_dir, output_dir)
    
    if results is not None:
        print("\n" + "="*80)
        print(" SAHI PARAMETER STUDY COMPLETE")
        print("="*80)
        print("Next steps:")
        print("1. Review optimal configurations in optimal_configurations.json")
        print("2. Use best configuration for interaction study")
        print("3. Include trade-off curves in paper figures")

if __name__ == '__main__':
    main()