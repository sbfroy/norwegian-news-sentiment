import numpy as np
import optuna
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.weighting import get_weight


def evaluate_score_ranges(predictions, targets):
    """RMSE broken down by target score range, to see where the model struggles."""
    ranges = [(-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1)]
    results = {}

    for i, (low, high) in enumerate(ranges):
        is_last = i == len(ranges) - 1
        # The last range includes its upper bound so that scores of exactly 1.0 are counted
        indices = [j for j, t in enumerate(targets) if low <= t < high or (is_last and t == high)]
        range_preds = [predictions[j] for j in indices]
        range_targets = [targets[j] for j in indices]

        if range_targets:
            results[f"{low} to {high}"] = np.sqrt(mean_squared_error(range_targets, range_preds))
        else:
            results[f"{low} to {high}"] = "No samples in range"

    return results


def train_model(model, train_dataset, val_dataset, optimizer, criterion, batch_size, num_epochs, device,
                trial=None, early_stopping=None, verbose=True):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    history = {
        'train_rmse': [],
        'val_rmse': [],
        'train_r2': [],
        'val_r2': [],
    }

    for epoch in range(num_epochs):

        # Training
        model.train()

        train_preds = []
        train_targets = []

        for batch in tqdm(train_loader, desc='train', leave=False, ncols=75, disable=not verbose):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs, attention_mask=masks).squeeze(-1)

            weights = torch.tensor(
                [get_weight(label.item()) for label in labels],
                dtype=torch.float,
                device=device
            )

            loss = criterion(outputs, labels)
            weighted_loss = (loss * weights).mean()
            weighted_loss.backward()
            optimizer.step()

            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        train_r2 = r2_score(train_targets, train_preds)

        # Validation
        model.eval()

        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='val', leave=False, ncols=75, disable=not verbose):
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, attention_mask=masks).squeeze(-1)

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)

        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f} | '
                  f'Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}')

            print("Validation RMSE by score range:")
            for score_range, rmse in evaluate_score_ranges(val_preds, val_targets).items():
                print(f"  {score_range}: {rmse}")

        # Optuna pruning
        if trial:
            trial.report(val_rmse, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stopping:
            early_stopping(val_rmse)
            if early_stopping.stop_training:
                print("Early stopping!")
                break

    return history
