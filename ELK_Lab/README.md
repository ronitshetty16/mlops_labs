# ELK Lab — Breast Cancer Classifier Monitoring

Pipeline that trains a **Gradient Boosting** classifier on the Breast Cancer Wisconsin dataset, detects data drift, and streams all logs to an **ELK Stack** (Elasticsearch + Logstash + Kibana) for real-time monitoring.

---

## What It Does

| Service | Role |
|---|---|
| `ml-trainer` | Retrains GradientBoostingClassifier every 2 min for 20 min, writes metrics to `training.log` |
| `drift-detector` | Simulates feature/schema drift every minute for 10 min, writes to `drift_detection.log` |
| `logstash` | Ingests both log files, parses metrics via grok, routes to Elasticsearch |
| `elasticsearch` | Indexes parsed events into `cancer-training-*` and `cancer-drift-*` indices |
| `kibana` | Visualizes metrics and drift events at `http://localhost:5601` |

---

## Dataset & Model

| Property | Details |
|---|---|
| Dataset | Breast Cancer Wisconsin (569 samples, 30 features) |
| Task | Binary classification: malignant (0) / benign (1) |
| Model | `GradientBoostingClassifier` (randomized hyperparams each run) |
| Preprocessing | `StandardScaler` + 5% Gaussian feature noise + 5% label noise |

**Logged training metrics:** Accuracy, F1, AUC-ROC, Precision, Recall, TP/TN/FP/FN, FPR, FNR

**Logged drift events:** batch number, F1, drift flag, drift type (`feature_shift` / `missing_features`)

---

## Project Structure

```
ELK_Lab/
├── docker-compose.yml      # All 5 services
├── Dockerfile.ml           # Image for ml-trainer and drift-detector
├── requirements.txt        # scikit-learn, numpy
├── train_model.py          # 20-min continuous retraining loop
├── drift_detection.py      # 10-min drift simulation loop
└── logstash/
    └── logstash.conf       # Dual-pipeline: training + drift → Elasticsearch
```

---

## Running the Project

```bash
docker compose up --build
```

This starts all 5 services. The ML containers start training/detecting immediately after Elasticsearch is healthy.

**View live Kibana:** `http://localhost:5601`

To stop and clean up:
```bash
docker-compose down -v
```

