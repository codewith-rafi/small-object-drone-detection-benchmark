#!/usr/bin/env python3
"""
Verify that all model backups exist and have correct sizes.
"""

import os
import hashlib

def get_file_info(path):
    """Get file size and MD5 hash."""
    if not os.path.exists(path):
        return None, None
    
    size = os.path.getsize(path)
    
    # Calculate MD5 hash
    md5_hash = hashlib.md5()
    with open(path, "rb") as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return size, md5_hash.hexdigest()

def main():
    print("="*80)
    print(" MODEL BACKUP VERIFICATION")
    print("="*80)
    
    # Define all backup locations
    locations = [
        ("Original best.pt", 
         "runs/detect/runs/safe/yolov8_xl_restart/weights/best.pt"),
        
        ("Primary backup", 
         "runs/detect/runs/safe/yolov8_xl_restart/weights/best_backup_epoch84.pt"),
        
        ("Secondary backup", 
         "best_model_backup/best_epoch84_verified.pt")
    ]
    
    print("\nChecking backup locations...")
    print("-"*80)
    
    file_infos = {}
    all_exist = True
    
    for name, path in locations:
        size, md5 = get_file_info(path)
        
        if size is None:
            print(f"❌ {name}: {path} - NOT FOUND")
            all_exist = False
        else:
            size_mb = size / (1024*1024)
            print(f"✅ {name}: {path}")
            print(f"   Size: {size_mb:.1f} MB ({size:,} bytes)")
            print(f"   MD5: {md5[:16]}...")
            file_infos[name] = (size, md5)
    
    print("\n" + "="*80)
    print(" VERIFICATION RESULTS")
    print("="*80)
    
    if not all_exist:
        print("❌ FAILED: Some backup files are missing")
        return
    
    # Check if all files have same size
    sizes = [info[0] for info in file_infos.values()]
    unique_sizes = set(sizes)
    
    if len(unique_sizes) == 1:
        print(f"✅ PASS: All files have same size: {sizes[0] / (1024*1024):.1f} MB")
    else:
        print("❌ FAIL: Files have different sizes!")
        for name, (size, _) in file_infos.items():
            print(f"   {name}: {size / (1024*1024):.1f} MB")
    
    # Check if all files have same MD5 (optional - can be slow for large files)
    print("\nChecking file integrity (MD5 comparison)...")
    md5_values = [info[1] for info in file_infos.values()]
    unique_md5 = set(md5_values)
    
    if len(unique_md5) == 1:
        print(f"✅ PASS: All files have identical content")
        print(f"   MD5: {md5_values[0]}")
    else:
        print("⚠️  WARNING: Files have different content!")
        for name, (_, md5) in file_infos.items():
            print(f"   {name}: {md5}")
    
    print("\n" + "="*80)
    print(" RECOMMENDED ACTIONS")
    print("="*80)
    
    if len(unique_sizes) == 1 and len(unique_md5) == 1:
        print("✅ All backups verified successfully!")
        print("\nNext steps:")
        print("1. Proceed with SAHI parameter study")
        print("2. Update professor contact template")
        print("3. Start P2 model training")
    else:
        print("⚠️  Backup verification issues detected")
        print("\nRecommended actions:")
        print("1. Re-create backups from original best.pt")
        print("2. Verify training results.csv for epoch 84 metrics")
        print("3. Test model inference before proceeding")

if __name__ == "__main__":
    main()