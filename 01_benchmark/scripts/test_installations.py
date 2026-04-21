#!/usr/bin/env python3
"""
Test script to verify all installations are working correctly.
"""

import sys
import importlib

def test_import(module_name, version_attr=None):
    """Test if a module can be imported and get its version."""
    try:
        module = importlib.import_module(module_name)
        if version_attr:
            version = getattr(module, version_attr, "Unknown")
        else:
            version = getattr(module, "__version__", "Unknown")
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("Testing installations...")
    print("=" * 50)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    
    # List of modules to test
    modules_to_test = [
        ("torch", "__version__"),
        ("torchvision", "__version__"),
        ("torchaudio", "__version__"),
        ("ultralytics", "__version__"),
        ("cv2", "__version__"),
        ("numpy", "__version__"),
        ("matplotlib", "__version__"),
        ("pandas", "__version__"),
        ("tqdm", "__version__"),
    ]
    
    all_passed = True
    
    for module_name, version_attr in modules_to_test:
        success, version = test_import(module_name, version_attr)
        status = "✓" if success else "✗"
        print(f"{status} {module_name}: {version}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    
    # Test CUDA availability for PyTorch
    if all_passed:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else "Not available"
            print(f"CUDA available: {cuda_available}")
            print(f"CUDA version: {cuda_version}")
            
            if cuda_available:
                print(f"GPU device: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except Exception as e:
            print(f"Error checking CUDA: {e}")
    
    # Test YOLO model loading
    if all_passed:
        try:
            from ultralytics import YOLO
            print("\nTesting YOLO model loading...")
            # Try to load a small model
            model = YOLO('yolov8n.pt')
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"✗ YOLO model loading failed: {e}")
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("All tests passed! ✅")
        return 0
    else:
        print("Some tests failed! ❌")
        return 1

if __name__ == "__main__":
    sys.exit(main())