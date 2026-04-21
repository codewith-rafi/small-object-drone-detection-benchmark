# VisDrone Dataset Analysis Report

## Executive Summary

The VisDrone dataset presents a challenging object detection task with **small objects** (66% of objects are <32px) and **high object density** (67 objects per image on average). The dataset is well-suited for benchmarking drone detection models, particularly for small object detection scenarios.

## Key Statistics

### Dataset Splits
| Split | Images | Total Objects | Avg Objects/Image | Avg Image Size | Small Objects % |
|-------|--------|---------------|-------------------|----------------|-----------------|
| Train | 1,000  | 67,741        | 67.74             | 1455×818       | 66.20%          |
| Val   | 548    | 35,251        | 64.33             | 1291×726       | 65.89%          |
| Test  | 209    | 13,446        | 64.33             | 1370×771       | 62.12%          |

### Class Distribution (Train Set)
| Class | Count | Percentage |
|-------|-------|------------|
| van   | 26,692 | 39.40%     |
| person| 21,070 | 31.10%     |
| bicycle| 8,626 | 12.73%     |
| truck | 3,337  | 4.93%      |
| car   | 3,225  | 4.76%      |
| tricycle | 2,022 | 2.98%      |
| awning-tricycle | 1,434 | 2.12%  |
| motor | 723    | 1.07%      |
| bus   | 612    | 0.90%      |

**Note:** Class distribution is highly imbalanced, with `van` and `person` dominating (70.5% combined).

## Critical Findings

### 1. Small Object Challenge
- **66.2%** of objects are smaller than 32×32 pixels (COCO small object definition)
- Average bounding box size: **34×34 pixels** (train set)
- This presents a significant challenge for standard object detectors

### 2. High Object Density
- Average of **67.7 objects per image** in training set
- Maximum observed: Likely much higher (analysis limited to 1000 images)
- Requires models with strong multi-object detection capabilities

### 3. Image Resolution Variability
- Average resolution: **~1.2 megapixels** (varies by split)
- Train: 1455×818 (1.19 MP)
- Val: 1291×726 (0.94 MP)
- Test: 1370×771 (1.06 MP)

### 4. Aspect Ratio Analysis
- Bounding boxes show varied aspect ratios
- Most objects have reasonable width/height ratios (0.5-2.0)
- Some extreme aspect ratios present (vehicles, people)

## Training Recommendations

### 1. Model Architecture
- **Use models with strong small object detection capabilities**
- Consider architectures with feature pyramid networks (FPN)
- YOLOv8/9/10 with appropriate neck designs recommended

### 2. Data Augmentation
- **Essential for small object detection:**
  - Mosaic augmentation (4-image mosaic)
  - MixUp augmentation
  - Random affine transformations
  - HSV color augmentation
  - Small object copy-paste augmentation

### 3. Input Resolution
- **Recommended:** 1280×1280 or higher
- Higher resolution helps with small object detection
- Balance with GPU memory constraints (RTX 5060 8GB)

### 4. Training Strategies
- **Class-balanced sampling** due to imbalanced distribution
- **Focal loss** for handling class imbalance
- **Longer training cycles** (300+ epochs)
- **Progressive resizing** (start smaller, increase resolution)

### 5. SAHI Implementation
- **Critical for inference:** Slicing Aided Hyper Inference
- Recommended slice size: 640×640 with 0.2 overlap
- Will significantly improve small object recall

## Benchmark Targets

Based on reference project (nia194/Object-Detection):
- **Target mAP@0.5:** >0.56 (YOLOv8-XL baseline)
- **Expected improvement with SAHI:** +10-20% mAP for small objects
- **Real-time target:** >30 FPS on RTX 5060

## Next Steps

1. **Baseline Training:** YOLOv8-XL with default settings
2. **SAHI Integration:** Implement and tune for VisDrone
3. **Advanced Models:** Benchmark YOLOv9-E, YOLOv10-X, YOLO26-X
4. **Optimization:** Hyperparameter tuning for small objects
5. **Evaluation:** Comprehensive metrics beyond mAP@0.5

## Risk Factors

1. **GPU Memory:** 8GB VRAM may limit batch size at high resolutions
2. **Training Time:** Large dataset (67K+ objects) requires significant compute
3. **Class Imbalance:** May require weighted loss functions
4. **Small Objects:** Standard evaluation metrics may not capture performance well

## Success Metrics

1. **Primary:** mAP@0.5 > 0.60 across all models
2. **Secondary:** Small object recall > 0.50
3. **Tertiary:** Inference speed > 30 FPS at 1280×1280
4. **Comparative:** Outperform reference project (0.56 mAP)

---

*Report generated: April 17, 2026*  
*Analysis based on 1,000 training images, 548 validation images, 209 test images*  
*Small object threshold: <32×32 pixels (COCO standard)*