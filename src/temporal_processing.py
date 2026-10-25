"""Published patient-window construction, with explicit token-level truncation.

The publication specifies selection ending at the last positive visit for cases,
or the last available visit for never-positive patients. All visits are chronological.
"""
import pandas as pd


def patient_windows(visits: pd.DataFrame, id_col='patient_id', date_col='visit_date',
                    text_col='note_text', label_col='apnea_label') -> pd.DataFrame:
    required = {id_col, date_col, text_col, label_col}
    if not required.issubset(visits):
        raise ValueError(f'Missing required columns: {sorted(required - set(visits))}')
    df = visits.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='raise')
    df[label_col] = pd.to_numeric(df[label_col], errors='raise').astype(int)
    if not set(df[label_col].unique()).issubset({0, 1}):
        raise ValueError('Visit labels must be binary 0/1')
    # stable tie handling depends on original row order; visit-order field if available is preferred.
    df = df.sort_values([id_col, date_col], kind='stable')
    out = []
    for patient_id, frame in df.groupby(id_col, sort=False):
        positives = frame[frame[label_col] == 1]
        label = int(not positives.empty)
        endpoint = positives.iloc[-1][date_col] if label else frame.iloc[-1][date_col]
        # positional slicing avoids including same-date visits AFTER the selected positive event
        end_pos = positives.index[-1] if label else frame.index[-1]
        # frame maintains original indices; get index position within sorted patient frame
        cut = frame.index.get_loc(end_pos)
        selected = frame.iloc[:cut + 1]
        out.append({'patient_id': patient_id, 'label': label,
                    'endpoint_date': endpoint,
                    'visits': [str(s) for s in selected[text_col].fillna('') if str(s).strip()]})
    return pd.DataFrame(out)


def tokenize_recent_history(visits, tokenizer, max_length=512, separator='\n'):
    """Keep rightmost tokens from chronological text, then add each model's specials.

    By default boundaries are encoded in chronological text with a newline; this
    approximates the recovered notebook. Selection is token-aware and preserves
    newest information. No artificial whitespace splitting is applied.
    """
    if max_length > tokenizer.model_max_length and tokenizer.model_max_length < 100000:
        raise ValueError('max_length exceeds model tokenizer limit')
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError('max_length is too small for model special tokens')
    chronological_text = separator.join(str(x) for x in visits if str(x).strip())
    # tokenize complete sequence and preserve latest subword tokens; no premature truncate.
    ids = tokenizer.encode(chronological_text, add_special_tokens=False, truncation=False)
    ids = ids[-capacity:]
    encoded = tokenizer.prepare_for_model(ids, add_special_tokens=True,
                                           max_length=max_length, truncation=False,
                                           padding='max_length', return_attention_mask=True)
    return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']}
