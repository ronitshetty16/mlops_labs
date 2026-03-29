import logging
import time
import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

logging.basicConfig(
    filename='/app/logs/drift_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_batch(X_scaled, y, batch_size=60, inject_drift=False):
    indices = np.random.choice(len(X_scaled), size=batch_size, replace=True)
    X_batch = X_scaled[indices].copy()
    y_batch = y[indices].copy()

    if not inject_drift:
        return X_batch, y_batch, False, None

    drift_count = max(1, int(0.1 * batch_size))
    drift_indices = np.random.choice(batch_size, size=drift_count, replace=False)
    drift_type = random.choice(["feature_shift", "missing_features"])

    for idx in drift_indices:
        if drift_type == "feature_shift":
            # Scale feature values far outside the training distribution
            X_batch[idx] = X_batch[idx] * np.random.uniform(2.0, 4.0)
            logger.warning(
                f"Drift Detected: True | Sample index: {idx} | "
                f"Drift Type: {drift_type}"
            )
        elif drift_type == "missing_features":
            # Zero out 5 random features (simulates missing/corrupted sensors)
            zero_feats = np.random.choice(X_batch.shape[1], size=5, replace=False)
            X_batch[idx, zero_feats] = 0.0
            logger.warning(
                f"Drift Detected: True | Sample index: {idx} | "
                f"Drift Type: {drift_type}"
            )

    return X_batch, y_batch, True, drift_type


def main():
    logger.info("Starting drift detection pipeline — Breast Cancer dataset")

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    start_time = time.time()
    batch = 1

    while time.time() - start_time < 600:  # 10 minutes
        inject_drift = random.random() < 0.3  # 30% chance per batch

        X_batch, y_batch, drift_occurred, drift_type = generate_batch(
            X_scaled, y, batch_size=60, inject_drift=inject_drift
        )

        logger.info(f"Batch: {batch} | Preprocessing batch of {len(X_batch)} samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X_batch, y_batch, test_size=0.3, random_state=42
        )

        try:
            model = GradientBoostingClassifier(
                n_estimators=random.choice([50, 100]),
                learning_rate=random.choice([0.05, 0.1]),
                max_depth=random.choice([2, 3])
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

            logger.info(
                f"Batch: {batch} | F1 Score: {f1:.4f} | "
                f"Drift Detected: {drift_occurred}"
            )

            if drift_occurred:
                logger.warning(
                    f"Batch: {batch} | Drift Type: {drift_type} | "
                    f"Possible performance degradation"
                )

        except Exception as e:
            logger.error(f"Batch {batch} failed: {e}")

        batch += 1
        time.sleep(60)  # one batch per minute

    logger.info("Drift detection loop completed after 10 minutes")


if __name__ == "__main__":
    main()
