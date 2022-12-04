import os

import click
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

UNPROCESSED_DATASET_NAME = 'data.csv'
PROCESSED_DATASET_TRAIN_NAME = 'data_processed.csv'

# if -1 then generates every time, otherwise fixed to this value
RANDOM_STATE_CONST = -1


@click.command('preprocessing_and_split')
@click.option('--input-dir', type=click.Path())
@click.option('--output-dir', type=click.Path())
def preprocessing_and_split(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(os.path.join(input_dir, UNPROCESSED_DATASET_NAME))

    simple_imputer = SimpleImputer(strategy='most_frequent')
    processed_data = pd.DataFrame(simple_imputer.fit_transform(data), columns=data.columns)
    processed_data.to_csv(os.path.join(output_dir, PROCESSED_DATASET_TRAIN_NAME), index=False)

    if RANDOM_STATE_CONST == -1:
        X_train, X_val, y_train, y_val = train_test_split(
            processed_data.drop(columns = 'condition'),
            processed_data['condition'],
            test_size=0.3
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            processed_data.drop(columns = 'condition'),
            processed_data['condition'],
            test_size=0.3,
            random_state=RANDOM_STATE_CONST
        )

    X_train.to_csv(os.path.join(output_dir, 'features_train.csv'), index=False)
    X_val.to_csv(os.path.join(output_dir, 'features_val.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'target_train.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'target_val.csv'), index=False)


if __name__ == '__main__':
    preprocessing_and_split()
