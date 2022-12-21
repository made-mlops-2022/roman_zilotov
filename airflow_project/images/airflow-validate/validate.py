import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@click.command('validate')
@click.option('--input-dir', type=click.Path())
@click.option('--input-model-dir', type=click.Path())
@click.option('--output-metrics-dir', type=click.Path())
def validate(input_dir, input_model_dir, output_metrics_dir):
    os.makedirs(output_metrics_dir, exist_ok=True)
    features_val = pd.read_csv(os.path.join(input_dir, 'features_val.csv'))
    target_val = pd.read_csv(os.path.join(input_dir, 'target_val.csv'))

    with open(os.path.join(input_model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    target_val_pred = model.predict(features_val)

    metrics = {}
    metrics['accuracy'] = accuracy_score(target_val, target_val_pred)
    metrics['f1_score'] = f1_score(target_val, target_val_pred)

    with open(os.path.join(output_metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    validate()
