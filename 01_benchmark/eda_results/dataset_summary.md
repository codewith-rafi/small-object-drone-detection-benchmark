# Dataset summary (EDA)

EDA characterization used a **1,000-image train subsample** plus full local val/test tooling splits.

| Split | Images | Objects | Small (&lt;32px) |
|-------|-------:|--------:|----------------:|
| Train (EDA subsample) | 1,000 | 67,741 | 66.20% |
| Val | 548 | 35,251 | 65.89% |
| Test (local) | 209 | 13,446 | 62.12% |

Official VisDrone-DET train contains **6,471** images. See `dataset_analysis_report.md` and `../results/baseline_metrics.json`.
