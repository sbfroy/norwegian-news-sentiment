"""Fine-tune NB-BERT with a regression head."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from transformers import AutoTokenizer

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import TokenizedDataset
from src.data.preprocessing import create_df
from src.models.transformer_model import TransformerModel
from src.training.train import train_model
from src.utils.config_loader import load_config
from src.utils.seed import seed_everything

PRETRAINED_NAME = 'NbAiLab/nb-bert-base'

config = load_config(base_dir / 'transformer_params.yaml')
seed_everything(config['general']['seed'])

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)

model = TransformerModel(
    transformer_name=PRETRAINED_NAME,
    dropout=config['model']['dropout']
)

# Load data
train_df = create_df(base_dir / 'data' / 'train')
val_df = create_df(base_dir / 'data' / 'dev')

train_dataset = TokenizedDataset(train_df, tokenizer, config['data']['max_seq_len'])
val_dataset = TokenizedDataset(val_df, tokenizer, config['data']['max_seq_len'])

# Weight decay is left at the AdamW default (0.01); the weight_decay in the config is not applied here
optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

criterion = nn.MSELoss()

history = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    criterion=criterion,
    batch_size=config['training']['batch_size'],
    num_epochs=config['training']['num_epochs'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Save config, training history and weights in a timestamped run folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = base_dir / 'logs' / f'run_{timestamp}'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'params.yaml', 'w') as f:
    yaml.dump(config, f)

pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)

torch.save(model.state_dict(), output_dir / 'transformer_model.pth')

print(f'Done! Saved run in {output_dir}')
