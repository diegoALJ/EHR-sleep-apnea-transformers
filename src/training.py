import random
import numpy as np
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .evaluation import classification_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['input_ids'], batch['attention_mask'])['logits']
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            labels.extend(batch['labels'].cpu().tolist())
    return classification_metrics(labels, probabilities), probabilities


def fit(model, train_set, val_set, *, epochs, batch_size, learning_rate,
        weight_decay, warmup_ratio, max_grad_norm, device):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * warmup_ratio), steps)
    best_score, best_state, history = -float('inf'), None, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, val_loader, device)
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), **metrics})
        score = metrics['roc_auc']
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
