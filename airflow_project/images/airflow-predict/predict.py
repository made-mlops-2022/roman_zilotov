import os
import click
import pickle
import pandas as pd

PROCESSED_DATASET_TRAIN_NAME = 'data_processed.csv'


@click.command('predict')
@click.option('--input-dir', type=click.Path())
@click.option('--input-model-dir', type=click.Path())
@click.option('--output-dir', type=click.Path())
def predict(input_dir, input_model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X = pd.read_csv(os.path.join(input_dir, PROCESSED_DATASET_TRAIN_NAME))
    with open(os.path.join(input_model_dir, 'model.pkl')) as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    pd.DataFrame(y_pred).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
        for elem in y_pred:
            f.writelines(f'{elem}\r\n')


if __name__ == '__main__':
    predict()
