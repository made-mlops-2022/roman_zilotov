import os
import pickle
import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RFC_N_ESTIMATORS = 120
RFC_MAX_DEPTH = 7


@click.command('train')
@click.option('--input-dir', type=click.Path())
@click.option('--output-dir', type=click.Path())
def train(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    X = pd.read_csv(os.path.join(input_dir, 'features_train.csv'))
    y = pd.read_csv(os.path.join(input_dir, 'target_train.csv'))

    model = RandomForestClassifier(n_estimators=RFC_N_ESTIMATORS, max_depth=RFC_MAX_DEPTH)
    model.fit(X, y)

    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
