import pickle
import click
import hydra
import pandas as pd
import logging
from sklearn.metrics import f1_score
from src import PATH_TO_SRC
from src.features import CustomTransformer
from hydra.core.config_store import ConfigStore
from src.entities import TrainPipelineCfg

logger = logging.getLogger("predict_pipeline")


def run_predict_pipeline(
        data_test: pd.DataFrame,
        target_true: pd.Series,
        prediction_output: str,
        param: TrainPipelineCfg
):
    transformer = CustomTransformer(
        data_test,
        param.feature_params.categorical,
        param.feature_params.numerical
    )
    transformer.fit()
    logger.info('Transformer successfully fitted.')
    data_test_processed = transformer.transform()
    logger.info('Data successfully transformed.')

    with open(PATH_TO_SRC.joinpath('models/model.pkl'), 'rb') as file:
        model = pickle.load(file)
    logger.info("Model loaded.")

    target_predict = model.predict(data_test_processed)
    logger.info("Target predicted.")
    logger.info(f"f1_score result: {f1_score(target_true, target_predict)}")

    with open(prediction_output, 'w') as f:
        for elem in target_predict:
            f.writelines(f'{elem}\r\n')
    logger.info(f"Predicted target was written to the {prediction_output}")


@click.command()
@click.option('--path_train_data', '-ptd',
              default=PATH_TO_SRC.joinpath('data/raw/features_test.csv'),
              help='Please enter path to file with your train data. Default path - data/raw/features_test.csv')
@click.option('--path_pred_target', '-ppt',
              default=PATH_TO_SRC.joinpath("data/predicted_target.txt"),
              help='Please enter path to file where you want save predicted target. Default path - '
                   'data/predicted_target.txt')

def main_wrapper(path_train_data: str, path_pred_target: str):
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainPipelineCfg)

    @hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
    def _main(param: TrainPipelineCfg):
        logger.info(f"Data for test was taken from #{path_train_data}")
        feature_test = pd.read_csv(path_train_data)
        target_true = pd.read_csv(PATH_TO_SRC.joinpath('data/raw/target_test.csv'))
        run_predict_pipeline(feature_test, target_true, path_pred_target, param)
    _main()


if __name__ == '__main__':
    main_wrapper()