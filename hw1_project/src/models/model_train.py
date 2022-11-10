import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from src.features import CustomTransformer
from src.entities import TrainPipelineCfg

DEFAULT_N_NEIGHBORS_FOR_GRID_SEARCH = [3, 5, 7, 11, 15]
DEFAULT_N_ESTIMATORS_FOR_GRID_SEARCH = [25, 50, 75, 100, 125, 150, 200]
DEFAULT_MAX_DEPTH_FOR_GRID_SEARCH = [2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_TRAIN_SIZE_FOR_GRID_SEARCH = 0.8

logger = logging.getLogger("train_pipeline -> train_model")

def train_model(
        X_train_processed: pd.DataFrame,
        X_train: pd.DataFrame,
        target: pd.Series,
        param: TrainPipelineCfg
) -> object:
    ##############
    # какие параметры включить?
    # max_depth
    # n_estimators
    # n разделить на n в knn и n в random_forest
    #
    #
    #
    #############
    if param.model.model == 'knn':
        model = KNeighborsClassifier(n_neighbors=param.model.n_neighbors)
        param_grid = {'n_neighbors': DEFAULT_N_NEIGHBORS_FOR_GRID_SEARCH,
                      'metric': ['minkowski', 'euclidean']}
    else:
        model = RandomForestClassifier(n_estimators=param.model.n_estimators,
                                       max_depth=param.model.max_depth,
                                       random_state=param.model.random_state)
        param_grid = {'n_estimators': DEFAULT_N_ESTIMATORS_FOR_GRID_SEARCH,
                      'max_depth': DEFAULT_MAX_DEPTH_FOR_GRID_SEARCH,
                      'max_features': ['sqrt', 'log2']}
    if param.model.grid_search:
        logger.info(f"grid_search deployed with parameters: {param_grid}")

        train_size = DEFAULT_TRAIN_SIZE_FOR_GRID_SEARCH
        split_rule = np.zeros(len(X_train))
        train_index = np.random.choice(
            range(len(X_train)),
            size=round(len(X_train) * train_size),
            replace=False
        )
        for i in train_index:
            split_rule[i] = -1

        ps = PredefinedSplit(split_rule)
        train_index, val_index = next(ps.split())
        data_train = X_train.iloc[train_index]
        data_val = X_train.iloc[val_index]

        transformer_train = CustomTransformer(
            data_train,
            param.feature_params.categorical,
            param.feature_params.numerical
        )
        transformer_train.fit()
        data_train_processed = transformer_train.transform()
        data_train_processed.set_index([pd.Index(train_index)], inplace=True)

        transformer_val = CustomTransformer(
            data_val,
            param.feature_params.categorical,
            param.feature_params.numerical
        )
        transformer_val.fit()
        data_val_processed = transformer_val.transform()
        data_val_processed.set_index([pd.Index(val_index)], inplace=True)

        X_val = pd.concat([data_train_processed, data_val_processed])
        X_val.sort_index(inplace=True)
        X_val.fillna(0, inplace=True)

        model_gs = GridSearchCV(estimator=model, param_grid=param_grid,
                                scoring='f1',
                                cv=PredefinedSplit(split_rule))
        model_gs.fit(X_val, target)
        logger.info(f"Chosen parameters: {model_gs.best_params_}")

        y_predict_train = model_gs.predict(X_train_processed)
        metrics = {'f1_score': f1_score(target, y_predict_train),
                   'accuracy_score': accuracy_score(target, y_predict_train),
                   'roc_auc_score': roc_auc_score(target, y_predict_train)}
        return metrics, model_gs



    logger.info('Grid search will not be deployed')
    model.fit(X_train_processed, target)
    y_predict_train = model.predict(X_train_processed)
    metrics = {'f1_score': f1_score(target, y_predict_train),
               'accuracy_score': accuracy_score(target, y_predict_train),
               'roc_auc_score': roc_auc_score(target, y_predict_train)}

    return metrics, model