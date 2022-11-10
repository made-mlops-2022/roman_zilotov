import hydra
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import train_model
from src import PATH_TO_SRC
from src.features import CustomTransformer
from hydra.core.config_store import ConfigStore
from src.entities import TrainPipelineCfg


PATH_TO_MODEL = PATH_TO_SRC.joinpath('models/')
logger = logging.getLogger("train_pipeline")


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def run_train_pipeline(param: TrainPipelineCfg):
    logger.info('run_train_pipeline started executing.')
    data = pd.read_csv(PATH_TO_SRC.joinpath('data/raw/heart_cleveland_upload.csv'))
    X = data.drop(columns='condition')
    y = data['condition']
    logger.info('Data successfully loaded.')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=param.splitting_params.test_size,
        random_state=param.splitting_params.random_state
    )
    logger.info(f'Data successfully splited with test_size = {param.splitting_params.test_size}'
                + f'and random_state = {param.splitting_params.random_state}.')

    data_transformer = CustomTransformer(
        X_train,
        param.feature_params.categorical,
        param.feature_params.numerical
    )
    data_transformer.fit()
    logger.info('Transformer successfully fitted.')

    X_train_transformed = data_transformer.transform()
    logger.info('Data successfully transformed.')

    y_train = y_train.reset_index(drop=True)
    data_transformed = pd.concat([X_train_transformed, pd.DataFrame(y_train, columns=['condition'])], axis=1)
    data_transformed.to_csv(PATH_TO_SRC.joinpath('data/processed/processed_train.csv'), index=False)
    logger.info('Transformed data successfully saved.')

    X_test.to_csv(PATH_TO_SRC.joinpath('data/raw/features_test.csv'), index=False)
    y_test.to_csv(PATH_TO_SRC.joinpath('data/raw/target_test.csv'), index=False)
    logger.info('Transformed data successfully saved.')

    metrics, model = train_model(X_train_transformed, X_train, y_train, param)
    logger.info('Model successfully trained.')
    logger.info(f"Method: {param.model.model}; f1_score on train data = {metrics['f1_score']}")

    with open(PATH_TO_MODEL.joinpath('model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    logger.info('Model successfully saved.')


def main():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainPipelineCfg)
    run_train_pipeline()


if __name__ == '__main__':
    main()