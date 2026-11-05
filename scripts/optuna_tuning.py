"""Hyperparameter search for the BiLSTM model with Optuna."""
import sys
from pathlib import Path

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import TokenizedDataset
from src.data.preprocessing import create_df
from src.models.BiLSTM_model import BiLSTM
from src.training.train import train_model
from src.utils.config_loader import load_config
from src.utils.seed import seed_everything

PRETRAINED_NAME = 'NbAiLab/nb-bert-base'

config = load_config(base_dir / 'BiLSTM_params.yaml')
seed_everything(config['general']['seed'])

optuna.logging.set_verbosity(optuna.logging.INFO)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
transformer = AutoModel.from_pretrained(PRETRAINED_NAME)

# Load data
train_df = create_df(base_dir / 'data' / 'train')
val_df = create_df(base_dir / 'data' / 'dev')

pretrained_embed = transformer.get_input_embeddings().weight.detach().cpu().numpy()


def objective(trial):

    # Architecture (hidden_size, num_layers) comes from the config; these are tuned
    dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 5e-7, 1e-5, log=True)

    train_dataset = TokenizedDataset(train_df, tokenizer, config['data']['max_seq_len'])
    val_dataset = TokenizedDataset(val_df, tokenizer, config['data']['max_seq_len'])

    model = BiLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_size=pretrained_embed.shape[1],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=dropout,
        pretrained_embed=pretrained_embed
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        trial=trial,
        verbose=False
    )

    return min(history['val_rmse'])


study = optuna.create_study(
    study_name='the_final_study',
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=config['general']['seed']),
    load_if_exists=True,
    storage='sqlite:///the_final_study.db'
)

study.optimize(objective, n_trials=50, show_progress_bar=True)
