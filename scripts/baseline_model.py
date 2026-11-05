"""Mean baseline: always predict the average training score."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.preprocessing import create_df

train_df = create_df(base_dir / 'data' / 'train')
val_df = create_df(base_dir / 'data' / 'dev')

mean_score = np.mean(train_df['score'].values)

val_labels = val_df['score'].values
val_predictions = [mean_score] * len(val_df)

rmse = np.sqrt(mean_squared_error(val_labels, val_predictions))

print(f"Baseline RMSE on Validation Set: {rmse:.4f}")  # roughly 0.45
