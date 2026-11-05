import torch.nn as nn
from transformers import AutoModel


class TransformerModel(nn.Module):
    """Pretrained transformer encoder with a linear regression head on the [CLS] token."""

    def __init__(self, transformer_name, dropout):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        self.dropout = nn.Dropout(p=dropout)
        self.regressor = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.regressor(cls_output)
