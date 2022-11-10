import pandas as pd
import matplotlib.pyplot as plt
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
from src import PATH_TO_SRC
from pathlib import Path


DESCRIPTION_PATH = 'tests/synthetic_data/description.json'
SYNTHETIC_DATA = 'tests/synthetic_data/synthetic_data.csv'
THRESHOLD = 7
EPS = 1
DEGREE_OF_BAYESIAN_NETWORK = 2
GEN_DATA_SIZE = 150


def main():
    input_data = Path(PATH_TO_SRC).joinpath('data/raw/heart_cleveland_upload.csv')
    categorical_attributes = {'sex': True, 'cp': True, 'fbs': True, 'restecg': True,
                              'exang': True, 'slope': True, 'ca': True, 'thal': True}

    describer = DataDescriber(category_threshold=THRESHOLD)
    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file=input_data,
        epsilon=EPS,
        k=DEGREE_OF_BAYESIAN_NETWORK,
        attribute_to_is_categorical=categorical_attributes
    )

    describer.save_dataset_description_to_file(DESCRIPTION_PATH)
    display_bayesian_network(describer.bayesian_network)

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(GEN_DATA_SIZE, DESCRIPTION_PATH)
    generator.save_synthetic_data(SYNTHETIC_DATA)

    synthetic_data = pd.read_csv(SYNTHETIC_DATA)
    df = pd.read_csv(PATH_TO_SRC.joinpath('data/raw/heart_cleveland_upload.csv'))

    attribute_description = read_json_file(DESCRIPTION_PATH)['attribute_description']
    inspector = ModelInspector(df, synthetic_data, attribute_description)
    inspector.mutual_information_heatmap()
    plt.savefig('tests/synthetic_data/heatmap.png')


if __name__ == '__main__':
    main()