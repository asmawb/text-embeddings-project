import numpy as np
from datasets import load_dataset
from .config import (
    DATASET_NAME,
    TEXT_COLUMN,
    LABEL_COLUMN,
    MAX_TRAIN_SAMPLES,
    MAX_TEST_SAMPLES,
    SEED,
)

def load_data():
    """
    Load AG News dataset and return train/test splits.
    """
    dataset = load_dataset(DATASET_NAME)

    train = dataset["train"].shuffle(seed=SEED)
    test = dataset["test"].shuffle(seed=SEED)

    if MAX_TRAIN_SAMPLES:
        train = train.select(range(min(len(train), MAX_TRAIN_SAMPLES)))
    if MAX_TEST_SAMPLES:
        test = test.select(range(min(len(test), MAX_TEST_SAMPLES)))

    X_train = np.array(train[TEXT_COLUMN])
    y_train = np.array(train[LABEL_COLUMN])

    X_test = np.array(test[TEXT_COLUMN])
    y_test = np.array(test[LABEL_COLUMN])

    return X_train, y_train, X_test, y_test
