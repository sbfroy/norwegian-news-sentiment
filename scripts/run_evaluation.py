"""Evaluate the trained BiLSTM model on the test set."""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.data.dataset import TokenizedDataset
from src.data.preprocessing import create_df
from src.models.BiLSTM_model import BiLSTM
from src.utils.config_loader import load_config

PRETRAINED_NAME = 'NbAiLab/nb-bert-base'

config = load_config(base_dir / 'BiLSTM_params.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
transformer = AutoModel.from_pretrained(PRETRAINED_NAME)

test_df = create_df(base_dir / 'data' / 'test')
test_dataset = TokenizedDataset(test_df, tokenizer, config['data']['max_seq_len'])
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

pretrained_embed = transformer.get_input_embeddings().weight.detach().cpu().numpy()

model = BiLSTM(
    vocab_size=tokenizer.vocab_size,
    embed_size=pretrained_embed.shape[1],
    hidden_size=config['model']['hidden_size'],
    num_layers=config['model']['num_layers'],
    dropout=config['model']['dropout'],
    pretrained_embed=pretrained_embed
)

weights_path = base_dir / 'src' / 'models' / 'BiLSTM_model.pth'
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(inputs, masks).squeeze(-1)

        test_preds.extend(outputs.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
test_r2 = r2_score(test_labels, test_preds)

print(f"Test RMSE: {test_rmse}")
print(f"Test R^2: {test_r2}")

output_dir = base_dir / 'evaluation_logs'
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'Test RMSE': float(test_rmse),
    'Test R^2': float(test_r2)
}

with open(output_dir / 'evaluation_results.yaml', 'w') as f:
    yaml.dump(results, f)

print(f'Done! Saved results in {output_dir}')
