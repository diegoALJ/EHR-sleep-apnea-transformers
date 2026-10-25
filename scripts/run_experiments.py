"""Run publication-inspired 3 backbones x 2 EHR representations.

Usage: python -m scripts.run_experiments --config configs/experiments.yaml
Requires permitted local patient-level data. No source EHR is distributed.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import PatientDataset
from src.models import EHRClassifier
from src.preprocessing import downsample_train, split_patients
from src.temporal_processing import patient_windows
from src.training import evaluate, fit, seed_everything
from src.tuning import grid_configs


def build_representation(raw, cfg, representation):
    cols = cfg['columns']
    text_column = cols['text'] if representation == 'full_ehr' else cols['entities']
    if text_column not in raw:
        raise ValueError(f"Missing {text_column}. For entities, provide precomputed, audited text; "
                         "automatic Spanish MedSpaCy rules are not historically validated.")
    values = raw[[cols['patient_id'], cols['visit_date'], text_column, cols['diagnosis']]].rename(
        columns={cols['patient_id']: 'patient_id', cols['visit_date']: 'visit_date',
                 text_column: 'note_text', cols['diagnosis']: 'apnea_label'})
    windows = patient_windows(values)
    if windows['visits'].map(len).eq(0).any():
        raise ValueError('Patient without content. Fix missing text/entity extraction before training.')
    return windows


def train_once(train, val, cfg, model_name, hp, output_path=None):
    seed_everything(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = PatientDataset(train, tokenizer, cfg['max_length'])
    val_ds = PatientDataset(val, tokenizer, cfg['max_length'])
    model = EHRClassifier(model_name, dropout=hp['dropout'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = fit(model, train_ds, val_ds, epochs=cfg['training']['epochs'],
                  batch_size=hp['batch_size'], learning_rate=hp['learning_rate'],
                  weight_decay=cfg['training']['weight_decay'],
                  warmup_ratio=cfg['training']['warmup_ratio'],
                  max_grad_norm=cfg['training']['max_grad_norm'], device=device)
    metrics, _ = evaluate(model, DataLoader(val_ds, batch_size=hp['batch_size']), device)
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.cpu().state_dict(), output_path / 'model_state.pt')
        tokenizer.save_pretrained(output_path / 'tokenizer')
        (output_path / 'history.json').write_text(json.dumps(history, indent=2))
        model.to(device)
    del train_ds, val_ds
    return model, tokenizer, metrics


def run(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(cfg['input_csv'])
    seed_everything(cfg['seed'])
    records = []
    for representation in cfg['representations']:
        windows = build_representation(raw, cfg, representation)
        train, val, test = split_patients(windows, **{
            'test_size': cfg['split']['test_size'], 'val_size': cfg['split']['val_size'],
            'seed': cfg['split']['random_state']})
        # Unlike the publication's described pre-split downsampling, do not change
        # the natural test/validation prevalence. This can change published metrics.
        train = downsample_train(train, cfg['balance']['target_positive_fraction'], cfg['seed'])
        best_hp = dict(cfg['training'])
        if cfg['tuning']['enabled']:
            search_results = []
            for idx, candidate in enumerate(grid_configs(cfg['tuning']['search_space'])):
                trial_model, trial_tokenizer, score = train_once(train, val, cfg, cfg['models'][cfg['tuning']['backbone']], candidate)
                del trial_model, trial_tokenizer
                search_results.append({'candidate': idx, **candidate, **score})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pd.DataFrame(search_results).to_csv(out / f'{representation}_grid.csv', index=False)
            metric = cfg['tuning']['metric']
            selected = max(search_results, key=lambda x: x[metric] if pd.notna(x[metric]) else -1)
            best_hp.update({k: selected[k] for k in cfg['tuning']['search_space']})
        for model_key, model_name in cfg['models'].items():
            run_dir = out / representation / model_key
            model, tok, val_metrics = train_once(train, val, cfg, model_name, best_hp, run_dir)
            device = next(model.parameters()).device
            test_ds = PatientDataset(test, tok, cfg['max_length'])
            test_metrics, probabilities = evaluate(model, DataLoader(test_ds, batch_size=best_hp['batch_size']), device)
            result = {'representation': representation, 'model': model_key,
                      'checkpoint': model_name, **{f'val_{k}': v for k, v in val_metrics.items()},
                      **{f'test_{k}': v for k, v in test_metrics.items()},
                      **{f'param_{k}': best_hp[k] for k in ('batch_size', 'dropout', 'learning_rate')}}
            records.append(result)
            pd.DataFrame({'patient_id': test['patient_id'], 'label': test['label'],
                          'probability': probabilities}).to_csv(run_dir / 'test_predictions_PRIVATE.csv', index=False)
            (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
            print(result)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(out / 'comparison.csv', index=False)
    print(f'Saved results in {out}. Do not publish patient-linked outputs.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/experiments.yaml')
    run(ap.parse_args().config)
