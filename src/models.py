"""BERT/RoBERTa compatible two-logit binary classification head."""
import torch
from torch import nn
from transformers import AutoModel


class EHRClassifier(nn.Module):
    def __init__(self, pretrained_name, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        result = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              return_dict=True)
        # BETO: historical pooler_output. RoBERTa lacks a consistently trained
        # sentence pooler, so use first-token final hidden state instead.
        model_type = getattr(self.encoder.config, 'model_type', '')
        if model_type == 'bert' and result.pooler_output is not None:
            pooled = result.pooler_output
        else:
            pooled = result.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def predict_probability(self, input_ids, attention_mask):
        self.eval()
        logits = self.forward(input_ids, attention_mask)['logits']
        return torch.softmax(logits, dim=-1)[:, 1]
