"""Patient-level data splits and training-only downsampling."""
import numpy as np
from sklearn.model_selection import train_test_split


def split_patients(frame, test_size=0.2, val_size=0.2, seed=42):
    if frame['patient_id'].duplicated().any():
        raise ValueError('Expected exactly one row per patient')
    train_dev, test = train_test_split(frame, test_size=test_size, stratify=frame['label'], random_state=seed)
    train, val = train_test_split(train_dev, test_size=val_size, stratify=train_dev['label'], random_state=seed)
    ids = [set(f.patient_id) for f in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def downsample_train(frame, positive_fraction=0.3, seed=42):
    """Reduce majority class only; do not resample validation or test.

    If positives are minority, retain all positives and sample negatives to the
    requested fraction; if negatives are minority, the method is symmetric.
    """
    if not 0 < positive_fraction < 1:
        raise ValueError('positive_fraction must be between zero and one')
    groups = {v: frame[frame.label == v] for v in (0, 1)}
    if min(map(len, groups.values())) == 0:
        raise ValueError('Both labels are needed for downsampling')
    # allow downsample of either class, never synthetic oversampling
    max_neg = int(len(groups[1]) * (1 - positive_fraction) / positive_fraction)
    if len(groups[0]) > max_neg:
        groups[0] = groups[0].sample(n=max_neg, random_state=seed)
    else:
        max_pos = int(len(groups[0]) * positive_fraction / (1 - positive_fraction))
        if len(groups[1]) > max_pos:
            groups[1] = groups[1].sample(n=max_pos, random_state=seed)
    if min(len(g) for g in groups.values()) < 1:
        raise ValueError('Not enough patients to achieve requested ratio')
    import pandas as pd
    return pd.concat(groups.values()).sample(frac=1, random_state=seed).reset_index(drop=True)
