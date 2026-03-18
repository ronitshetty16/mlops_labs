# Docker Lab 2 — Wine Cultivar Classifier

A containerized MLOps pipeline that trains a **Random Forest** classifier on the UCI Wine dataset and serves predictions through a **Flask** web API with a modern UI.

---

## What It Does

- **Trains** a scikit-learn Random Forest model on 13 chemical properties of wine
- **Predicts** one of three wine cultivar classes (`class_0`, `class_1`, `class_2`)
- **Serves** predictions via a Flask web app with probability bars for each class

---

## Project Structure

```
Docker_Lab2/
├── dockerfile              # Multi-stage build: train → serve
├── docker-compose.yml      # Two-service orchestration with shared volume
├── requirements.txt        # Python dependencies
└── src/
    ├── model_training.py   # Trains RandomForest, saves wine_model.pkl
    ├── main.py             # Flask app (port 4000)
    └── templates/
        └── predict.html    # Frontend UI
```

---

## Dataset & Model

| Property | Details |
|---|---|
| Dataset | [UCI Wine](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) (178 samples) |
| Features | 13 chemical measurements (alcohol, malic acid, ash, etc.) |
| Target | 3 cultivar classes |
| Model | `RandomForestClassifier` (100 trees, max_depth=10) |
| Preprocessing | `StandardScaler` normalization |
| Artifact | `wine_model.pkl` (model + scaler bundled) |

---

## Running the Project

### Option A — Docker Compose (recommended, mirrors the lab architecture)

Spins up two containers: one trains the model, the other serves it. The trained model is passed between them via a shared Docker volume.

```bash
docker-compose up
```

Open `http://localhost` in your browser.

To stop:
```bash
docker-compose down
```

---

### Option B — Single Multi-Stage Image

Trains the model at build time (Stage 1) and bakes the artifact into the serving image (Stage 2).

```bash
docker build -t wine-classifier .
docker run -p 80:4000 wine-classifier
```

Open `http://localhost` in your browser.

---

## Docker Architecture

### Docker Compose (Option A)

```
┌─────────────────────┐        shared volume        ┌──────────────────────┐
│   model-training    │  ──── wine_model.pkl ───►   │      serving         │
│  python:3.11-slim   │       /exchange/             │  python:3.11-slim    │
│                     │                              │  Flask on port 4000  │
│  Trains RF model    │                              │  Host port 80        │
└─────────────────────┘                              └──────────────────────┘
```

### Multi-Stage Dockerfile (Option B)

```
Stage 1 (trainer)          Stage 2 (server)
─────────────────          ────────────────
Install deps               Install deps
Run model_training.py  ──► COPY wine_model.pkl
Save wine_model.pkl        COPY main.py + templates
                           EXPOSE 4000
                           CMD python main.py
```

---

## Using the Web UI

1. Open `http://localhost`
2. Click **Load Example** to auto-fill sample wine measurements
3. Click **Predict** to get the cultivar class and per-class probabilities
4. Use **Clear** to reset the form

---

## Comparison with Lab 2 (Iris)

| | Lab 2 (Iris) | This Project (Wine) |
|---|---|---|
| Dataset | Iris (4 features) | Wine (13 features) |
| Model | TensorFlow Neural Network | scikit-learn Random Forest |
| Model format | `.keras` | `.pkl` |
| Framework | TensorFlow / Keras | scikit-learn |
| Frontend style | Cyberpunk / glassmorphism | Clean modern UI |
| Base image | `python:3.10` | `python:3.11-slim` |
