from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
import unittest
from sklearn.model_selection import train_test_split
from src.features import CustomTransformer
from src.entities.train_pipeline_params import SplittingParams, FeatureParams, ModelParams, TrainPipelineCfg
from src.models import train_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical = ['sex', 'cp', 'fbs', 'restecg',
                            'exang', 'slope', 'ca', 'thal']
        self.numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.target_col = 'condition'

        self.data = pd.read_csv('tests/synthetic_data/synthetic_data.csv')
        self.X = self.data.drop(columns=[self.target_col])
        self.target = self.data[self.target_col]
        self.split_param = SplittingParams()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.target,
            test_size=self.split_param.test_size,
            random_state=self.split_param.random_state
        )

        self.feat_param = FeatureParams(
            categorical=self.categorical,
            numerical=self.numerical,
            target_col=self.target_col
        )

        transformer_train = CustomTransformer(
            self.X_train,
            self.categorical,
            self.numerical
        )

        transformer_train.fit()

        self.X_train_processed = transformer_train.transform()
        transformer_test = CustomTransformer(
            self.X_test,
            self.categorical,
            self.numerical
        )
        transformer_test.fit()
        self.X_test_processed = transformer_test.transform()
        train_config = TrainPipelineCfg(
            model=ModelParams(
                [3, 5, 7, 11, 15]
            ),
            splitting_params=self.split_param,
            feature_params=self.feat_param
        )
        self.metrics, self.model = train_model(self.X_train_processed, self.X_train, self.y_train, train_config)

    def test_transformer(self):
        expect = (len(self.X) * (1 - self.split_param.test_size), 13)
        expect_processed = (len(self.X) * (1 - self.split_param.test_size), 28)
        self.assertEqual(self.X_train.shape, expect)
        self.assertEqual(self.X_train_processed.shape, expect_processed)
        expect = (len(self.X) * self.split_param.test_size, 13)
        expect_processed = (len(self.X) * self.split_param.test_size, 28)
        self.assertEqual(self.X_test.shape, expect)
        self.assertEqual(self.X_test_processed.shape, expect_processed)
        self.assertIsInstance(self.X_train_processed, pd.DataFrame)
        self.assertIsInstance(self.X_test_processed, pd.DataFrame)

    def test_fit_pipeline(self):
        self.assertEqual(len(self.y_train), len(self.X) * (1 - self.split_param.test_size))
        tmp = list(np.unique(self.y_train))
        self.assertEqual(tmp, [0, 1])
        self.assertGreaterEqual(self.metrics['f1_score'], 0.6)
        check_is_fitted(self.model)

    def test_predict_pipeline(self):
        y_predicted = self.model.predict(self.X_test_processed)
        self.assertIsInstance(y_predicted, np.ndarray)
        self.assertEqual(len(y_predicted), len(self.X) * self.split_param.test_size)
        self.assertGreaterEqual(f1_score(self.y_test, y_predicted), 0.5)
        tmp = list(np.unique(self.y_train))
        self.assertEqual(tmp, [0, 1])

