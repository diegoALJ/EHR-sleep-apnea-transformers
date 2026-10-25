import pandas as pd
from src.temporal_processing import patient_windows, tokenize_recent_history
from src.preprocessing import split_patients, downsample_train


class FakeTokenizer:
    model_max_length = 512
    def num_special_tokens_to_add(self, pair=False):
        return 2
    def encode(self, text, add_special_tokens=False, truncation=False):
        return [ord(c) for c in text]
    def prepare_for_model(self, ids, add_special_tokens=True, max_length=512,
                          truncation=False, padding='max_length', return_attention_mask=True):
        out = [101] + ids + [102]
        mask = [1] * len(out)
        return {'input_ids': out + [0] * (max_length-len(out)),
                'attention_mask': mask + [0] * (max_length-len(mask))}


def test_last_positive_visit_is_endpoint():
    f = pd.DataFrame({'patient_id':[1, 1, 1, 2, 2],
                      'visit_date':['2022-01-01','2022-03-01','2022-02-01','2022-01-01','2022-02-01'],
                      'note_text':['old','after','positive','other','last'],
                      'apnea_label':[0,0,1,0,0]})
    result = patient_windows(f).set_index('patient_id')
    assert result.loc[1, 'visits'] == ['old', 'positive']
    assert result.loc[1, 'label'] == 1
    assert result.loc[2, 'visits'] == ['other', 'last']


def test_recent_token_truncation_preserves_end_of_sequence():
    tok = tokenize_recent_history(['abcdef', 'uvwxyz'], FakeTokenizer(), max_length=8)
    assert tok['input_ids'] == [101] + [ord(x) for x in 'uvwxyz'] + [102]


def test_patient_splits_disjoint():
    frame = pd.DataFrame({'patient_id': list(range(100)),
                          'label': [i % 2 for i in range(100)]})
    train,val,test = split_patients(frame)
    assert not (set(train.patient_id) & set(val.patient_id))
    assert not (set(train.patient_id) & set(test.patient_id))
    assert not (set(val.patient_id) & set(test.patient_id))
    sampled = downsample_train(train, positive_fraction=.3)
    assert not (set(sampled.patient_id) - set(train.patient_id))
