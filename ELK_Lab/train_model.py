import logging
import time
import random
import numpy as np
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)

logging.basicConfig(
    filename='/app/logs/training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_and_evaluate():
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Add small Gaussian feature noise to simulate variability
    noise = np.random.normal(0, 0.05, X.shape)
    X_noisy = X + noise

    # Inject 5% label noise for robustness testing
    y_noisy = y.copy()
    noise_idx = np.random.choice(len(y), size=int(0.05 * len(y)), replace=False)
    y_noisy[noise_idx] = 1 - y_noisy[noise_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y_noisy, test_size=0.2, random_state=random.randint(0, 100)
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Randomized hyperparameters
    n_estimators = random.choice([50, 100, 150, 200])
    learning_rate = random.choice([0.01, 0.05, 0.1, 0.2])
    max_depth = random.choice([2, 3, 4, 5])

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    auc_roc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    logger.info("=== Training Run ===")
    logger.info(
        f"Hyperparameters: n_estimators={n_estimators}, "
        f"learning_rate={learning_rate}, max_depth={max_depth}"
    )
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    logger.info(f"FPR: {fpr:.4f}")
    logger.info(f"FNR: {fnr:.4f}")


def main():
    logger.info("Starting Breast Cancer GradientBoosting training loop")
    start_time = time.time()
    run = 1

    while time.time() - start_time < 1200:  # 20 minutes
        try:
            logger.info(f"--- Run {run} started at {datetime.now().isoformat()} ---")
            train_and_evaluate()
            logger.info(f"--- Run {run} completed ---")
            run += 1
        except Exception as e:
            logger.error(f"Training run {run} failed: {e}")

        time.sleep(120)  # retrain every 2 minutes

    logger.info("Training loop completed after 20 minutes")


if __name__ == "__main__":
    main()
