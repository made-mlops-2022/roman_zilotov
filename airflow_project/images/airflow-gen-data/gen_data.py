import os
import click
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network

INPUT_DATASET_NAME = 'origin_data.csv'
OUTPUT_DATASET_NAME = 'data.csv'
OUTPUT_DATASET_DESCRIPTION_NAME = 'description.json'
THRESHOLD = 5
EPS = 1
DEGREE_OF_BAYESIAN_NETWORK = 2
GEN_DATA_SIZE = 200
CATEGORICAL_ATTRIBUTES = {
    'sex': True,
    'cp': True,
    'fbs': True,
    'restecg': True,
    'exang': True,
    'ca': True,
    'thal': True
}


@click.command('gen')
@click.option('--output-dir', type=click.Path())
def gen_data(output_dir) -> None:
    os.makedirs(output_dir, exist_ok=True)

    path_to_output_data = os.path.join(output_dir, OUTPUT_DATASET_NAME)

    describer = DataDescriber(category_threshold=THRESHOLD)
    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file=INPUT_DATASET_NAME,
        epsilon=EPS,
        k=DEGREE_OF_BAYESIAN_NETWORK,
        attribute_to_is_categorical=CATEGORICAL_ATTRIBUTES
    )
    describer.save_dataset_description_to_file(OUTPUT_DATASET_DESCRIPTION_NAME)
    display_bayesian_network(describer.bayesian_network)
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(GEN_DATA_SIZE, OUTPUT_DATASET_DESCRIPTION_NAME)
    generator.save_synthetic_data(path_to_output_data)
    os.remove(OUTPUT_DATASET_DESCRIPTION_NAME)


if __name__ == '__main__':
    gen_data()