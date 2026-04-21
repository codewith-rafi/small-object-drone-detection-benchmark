#!/usr/bin/env python3
"""
Quick validation of best.pt model to verify it corresponds to epoch 84.
"""

from ultralytics import YOLO
import os

def main():
    print("="*80)
    print(" VALIDATING best.pt MODEL (Epoch 84)")
    print("="*80)
    
    # Paths
    model_path = 'runs/safe/yolov8_xl_restart/weights/best.pt'
    data_yaml = '../data/visdrone_simple.yaml'
    
    print(f"\n📁 Model: {model_path}")
    print(f"📊 Dataset: {data_yaml}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset config not found: {data_yaml}")
        return
    
    print("\n🔍 Loading model...")
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Model size: {os.path.getsize(model_path) / 1e6:.1f} MB")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n📈 Running validation...")
    print("   This may take 5-10 minutes...")
    
    try:
        # Run validation with minimal settings for speed
        results = model.val(
            data=data_yaml,
            batch=1,
            imgsz=1280,
            device='0',
            workers=0,
            verbose=False
        )
        
        print("\n✅ VALIDATION COMPLETE")
        print("="*80)
        
        # Extract key metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"   mAP@0.5:     {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"   mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"   Precision:   {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"   Recall:      {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
            
            # Compare with expected values from epoch 84
            print(f"\n🔍 COMPARISON WITH EPOCH 84:")
            print(f"   Expected mAP@0.5: 0.5375")
            print(f"   Actual mAP@0.5:   {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            
            actual_map = metrics.get('metrics/mAP50(B)', 0)
            if abs(actual_map - 0.5375) < 0.01:
                print(f"   ✅ VERIFIED: Model matches epoch 84 performance")
            else:
                print(f"   ⚠️  WARNING: Model performance differs from epoch 84")
                print(f"      Difference: {actual_map - 0.5375:+.4f}")
        
        else:
            print("❌ No metrics returned from validation")
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print(" NEXT STEPS RECOMMENDATION")
    print("="*80)
    print("1. If mAP@0.5 ≈ 0.5375 → Proceed with SAHI study")
    print("2. Update professor contact with results")
    print("3. Start P2 model training")
    print("4. Prepare interaction study")

if __name__ == '__main__':
    main()