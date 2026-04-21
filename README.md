# Drone Detection Benchmark

## Project Overview
Research project benchmarking YOLO models on the VisDrone dataset for small object detection in aerial imagery. This project focuses on the challenging task of detecting small objects (66% < 32px) with high density (67 objects per image).

## Key Results
- **Baseline Performance**: YOLOv8-XL achieved **0.5375 mAP@0.5** (96% of target 0.56)
- **Dataset**: VisDrone with 66% small objects (<32px)
- **Training Time**: ~50.1 hours on RTX 5060 8GB
- **Best Model**: Saved at epoch 84 before training crash at epoch 88
- **Research Focus**: Interaction study between SAHI (Slicing Aided Hyper Inference) and P2 head modifications

## Dataset
- **Name**: VisDrone Dataset (aerial imagery)
- **Classes**: 9 classes (person, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- **Statistics**: 
  - Train: 1,000 images, 67,741 objects
  - Val: 548 images, 35,251 objects
  - Test: 209 images, 13,446 objects
- **Challenge**: 66% of objects are smaller than 32×32 pixels

## Setup Instructions

### 1. Environment Setup
```bash
# Using Conda
conda env create -f environment.yml
conda activate drone_detection

# Or using pip
pip install -r requirements.txt
```

### 2. Dataset Preparation
1. Download VisDrone dataset from [VisDrone Official Website](http://aiskyeye.com/)
2. Extract to `01_benchmark/data/visdrone_raw/`
3. Convert to YOLO format:
```bash
python 01_benchmark/scripts/visdrone2yolo.py --create-yaml
```

### 3. Environment Verification
```bash
python 01_benchmark/scripts/test_installations.py
```

## Usage

### Training
```bash
# Interactive training launcher (Windows)
01_benchmark\start_training.bat

# Direct training
python 01_benchmark/scripts/train_yolov8_xl.py

# Monitor training progress
python 01_benchmark/scripts/monitor_training.py
```

### Analysis
```bash
# Exploratory Data Analysis
python 01_benchmark/scripts/eda_visdrone.py

# Validate model
python 01_benchmark/scripts/validate_best_model.py

# Verify backups
python 01_benchmark/scripts/verify_backups.py
```

### Research
```bash
# SAHI parameter study
python 01_benchmark/scripts/sahi_parameter_study.py

# P2 head modifications
python 01_benchmark/scripts/p2_head_modification.py

# Interaction metrics
python 01_benchmark/scripts/interaction_metrics.py
```

## Research Plan

### Phase 1: Baseline (Completed)
- ✅ YOLOv8-XL training: 0.5375 mAP@0.5
- ✅ Dataset analysis and EDA
- ✅ Model verification and backup

### Phase 2: SAHI Parameter Study (Next)
- SAHI optimization for small object detection
- Grid search for slice sizes and overlap ratios
- Expected improvement: 5-15% mAP

### Phase 3: P2 Model Training
- P2 head modification implementation
- Training P2-modified YOLOv8-XL
- Expected improvement: 5-10% for small objects

### Phase 4: Interaction Study
- 4-condition ablation study:
  1. Baseline (0.5375 mAP)
  2. Baseline + SAHI
  3. P2-modified model
  4. P2 + SAHI (interaction)
- Synergy and redundancy analysis

## File Structure
```
drone-detection-benchmark/
├── 01_benchmark/
│   ├── scripts/                    # Python scripts
│   │   ├── eda_visdrone.py         # EDA analysis
│   │   ├── train_yolov8_xl.py      # Main training script (Windows-safe)
│   │   ├── verify_backups.py       # Backup verification
│   │   ├── validate_best_model.py  # Model validation
│   │   ├── sahi_parameter_study.py # SAHI research
│   │   ├── p2_head_modification.py # P2 modifications
│   │   └── ... (see full list below)
│   ├── eda_results/                # EDA reports and visualizations
│   ├── data/                       # Dataset configurations
│   │   ├── visdrone.yaml          # Complete config
│   │   └── visdrone_simple.yaml   # Simplified config
│   └── start_training.bat         # Training launcher
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Scripts Overview

### Core Scripts
- `eda_visdrone.py` - Comprehensive dataset analysis
- `train_yolov8_xl.py` - Windows-safe YOLOv8-XL training (batch=1, workers=0)
- `visdrone2yolo.py` - Dataset format conversion
- `test_installations.py` - Environment verification

### Verification & Validation
- `verify_backups.py` - Model backup integrity check
- `validate_best_model.py` - Model performance validation
- `monitor_training.py` - Training progress monitoring

### Research Scripts
- `sahi_parameter_study.py` - SAHI parameter optimization
- `p2_head_modification.py` - P2 architecture modifications
- `interaction_metrics.py` - Interaction study metrics
- `failure_analysis.py` - Failure case analysis
- `check_architecture.py` - Model architecture verification

### Documentation
- `training_summary.md` - Baseline training results
- `what_happened_summary.md` - Training timeline
- `next_steps_plan.md` - Research roadmap
- `interaction_study_plan.md` - Interaction study plan

## Hardware Requirements
- **GPU**: NVIDIA RTX 5060 8GB (tested)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ for dataset and models
- **OS**: Windows 10/11 (scripts include Windows-specific fixes)

## Configuration Notes
- **Batch Size**: 1 (optimized for 8GB VRAM on Windows)
- **Workers**: 0 (prevents dataloader deadlock on Windows)
- **Image Size**: 1280×1280 (optimized for small object detection)
- **Augmentations**: Mosaic, copy-paste, HSV (critical for small objects)

## Model Weights
Model weights (`.pt` files) are excluded from this repository due to size (4.89 GB total). To obtain trained models:
1. Train from scratch using provided scripts
2. Contact maintainer for model download links
3. Use pre-trained YOLO weights from Ultralytics

## License
This project is for research purposes. Dataset usage must comply with VisDrone license terms.

## Citation
If you use this work in your research, please cite:
```bibtex
@software{drone_detection_benchmark_2026,
  title = {Drone Detection Benchmark: SAHI-P2 Interaction Study},
  author = {Rafi},
  year = {2026},
  url = {https://github.com/codewith-rafi/drone-detection-benchmark}
}
```

## Contact
For questions or collaboration, please open an issue on GitHub or contact the maintainer.

## Acknowledgments
- VisDrone dataset creators
- Ultralytics YOLO team
- SAHI library developers