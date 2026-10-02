# Interaction Study Execution Plan
## SAHI + P2 Head Interaction for Small Object Detection

### Current Status
- **Baseline training**: YOLOv8-XL running (epoch 5/150, mAP@0.5=0.299)
- **Expected completion**: ~4 days (150 epochs or early stop)
- **Current progress**: ~3.3% complete

### Phase 1: Baseline Completion & Analysis (Current)
1. **Monitor baseline training** (auto, ~4 days)
   - Checkpoint: best.pt, last.pt every 20 epochs
   - Monitor metrics: mAP@0.5, mAP@0.5:0.95, precision, recall
   - Target: Match/reference 0.56 mAP@0.5

2. **Baseline analysis** (after training)
   - Final performance metrics
   - Small object detection analysis (objects < 32px)
   - Error analysis: false positives, false negatives

### Phase 2: SAHI Parameter Study (After Baseline)
1. **Run SAHI parameter sweep** (1-2 hours)
   - Script: `sahi_parameter_study.py`
   - Grid: 4 slice sizes × 4 overlap ratios = 16 configurations
   - Output: Optimal configurations for detection vs speed

2. **Select SAHI configurations for interaction study**
   - Best detection configuration (max accuracy)
   - Best trade-off configuration (balanced)
   - Use in Phase 3 experiments

### Phase 3: P2 Model Training (Parallel with SAHI Study)
1. **Train P2-modified YOLOv8-XL** (~4 days)
   - Configuration: `p2_configs/yolov8x_p2.yaml`
   - Settings: Same as baseline (batch=1, imgsz=1280, epochs=150)
   - Expected: ~15-20% more VRAM usage
   - Monitor: Small object detection improvement

2. **P2 model analysis**
   - Compare with baseline
   - Small object detection improvement
   - Computational cost analysis

### Phase 4: Interaction Study (Core Contribution)
1. **Four experimental conditions**:
   - **Condition A**: Baseline (original YOLOv8-XL)
   - **Condition B**: Baseline + SAHI (optimal configuration)
   - **Condition C**: P2-modified model (no SAHI)
   - **Condition D**: P2-modified model + SAHI

2. **Evaluation protocol**:
   - Use same validation set (548 images)
   - Same evaluation metrics
   - Focus on small objects (< 32px)
   - Measure inference time/FPS

3. **Interaction metrics calculation**:
   - Script: `interaction_metrics.py`
   - Synergy index: S = (D - C) - (B - A)
   - Redundancy score: R = 1 - (D - A) / ((B - A) + (C - A))
   - Interaction gain: IG = (D - A) - max(B - A, C - A)

### Phase 5: Failure Analysis & Practical Guidelines
1. **Failure case analysis**:
   - Script: `failure_analysis.py`
   - When does SAHI fail? (over-segmentation, duplicate detections)
   - When does P2 fail? (noise amplification, overfitting)
   - Combined failure modes

2. **Practical guidelines**:
   - When to use SAHI vs P2 vs both
   - Resource-accuracy trade-offs
   - Deployment recommendations

### Phase 6: Paper Preparation
1. **Results compilation** (2-3 days)
   - Tables: Performance comparison
   - Figures: Trade-off curves, heatmaps, failure cases
   - Statistical analysis

2. **Professor contact** (epoch 20-30 of baseline)
   - Preliminary results
   - Research direction confirmation
   - Collaboration discussion

3. **Paper writing** (1 month)
   - Abstract, introduction, methodology
   - Results, discussion, conclusion
   - Target: MDPI Drones journal

### Timeline Estimate
```
Week 1 (Current): Baseline training completion
Week 2: SAHI study + P2 training
Week 3: Interaction study execution
Week 4: Analysis & professor contact
Month 2: Paper writing
Month 3: Submission preparation
```

### Resource Requirements
- **GPU**: RTX 5060 (8GB VRAM) - sufficient for all experiments
- **Storage**: ~50GB for models, results, visualizations
- **Time**: ~3 weeks for experiments, 1 month for paper

### Risk Mitigation
1. **Training failure**: Checkpoint every 20 epochs, early stopping
2. **VRAM issues**: Reduce batch size, use AMP, monitor usage
3. **Poor results**: Analyze failures, adjust hyperparameters
4. **Time constraints**: Focus on core interaction study, simplify comparisons

### Success Criteria
1. **Technical**: Demonstrate SAHI+P2 interaction (synergy/redundancy)
2. **Academic**: Publish in MDPI Drones (Q1/Q2 journal)
3. **Practical**: Provide guidelines for drone detection practitioners

### Next Immediate Actions
1. Continue monitoring baseline training
2. Prepare professor contact email template
3. Test SAHI script on small subset
4. Verify P2 configuration works with Ultralytics