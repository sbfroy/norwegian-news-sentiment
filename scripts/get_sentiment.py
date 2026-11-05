"""Score news articles from Norsk Aviskorpus with the fine-tuned NB-BERT model."""
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.models.transformer_model import TransformerModel
from src.utils.config_loader import load_config

PRETRAINED_NAME = 'NbAiLab/nb-bert-base'

config = load_config(base_dir / 'transformer_params.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)

model = TransformerModel(
    transformer_name=PRETRAINED_NAME,
    dropout=config['model']['dropout']
)

weights_path = base_dir / 'src' / 'models' / 'transformer_model.pth'
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.to(device)
model.eval()


def get_sentiment(text):
    """Return a sentiment score in roughly [-1, 1] for a single text."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config['data']['max_seq_len'],
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    return output.item()


input_path = base_dir / 'data' / 'all_articles_from_norsk_aviskorpus.csv'
output_path = base_dir / 'data' / 'all_articles_with_sentiment.csv'

data = pd.read_csv(input_path)

tqdm.pandas()
data['score'] = data['text'].progress_apply(get_sentiment)

data[['id', 'date', 'paper', 'score']].to_csv(output_path, index=False)

print(f"Done! Results saved in {output_path}")
