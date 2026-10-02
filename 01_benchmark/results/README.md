# Public Results Artifacts

Aggregate metrics only. Model weights (`.pt`) and VisDrone images are intentionally excluded.

| File | Description |
|------|-------------|
| `baseline_metrics.json` | Sealed YOLOv8-XL baseline summary (best epoch 84) |
| `yolov8x_baseline_results.csv` | Full per-epoch Ultralytics validation log from the Windows-safe run |
| `training_curves.png` | mAP@0.5 and train-loss curves derived from the CSV |

## Provenance

Source run (local only, not in git):

`01_benchmark/scripts/runs/detect/runs/safe/yolov8_xl_restart/results.csv`

Best validation mAP@0.5 = **0.53746** at epoch **84** (mAP@0.5:0.95 = 0.34715).
Training stopped after NaN losses (epochs 85–87) and a metric reset at epoch 88.

## Regenerating plots

```bash
python -c "import json; print(json.load(open('01_benchmark/results/baseline_metrics.json'))['best'])"
```

To regenerate `training_curves.png` after updating the CSV, re-plot `metrics/mAP50(B)` and the three train loss columns against `epoch`.

## Not included yet

SAHI parameter grids, P2-head trained models, and SAHI×P2 interaction metrics are **protocol-only** in this release. See `01_benchmark/scripts/sahi_parameter_study.py` and `interaction_study_plan.md`.
