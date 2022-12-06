import pandas as pd
import numpy as np
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, X: pd.DataFrame, categorical_features: List[str], numerical_features: List[str]):
        self.X = X.copy()
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def process_categorical_features(self, categorical_data: pd.DataFrame) -> pd.DataFrame:
        categorical_pipeline = self.build_categorical_pipeline()
        categorical_pipeline.fit(categorical_data)
        data_pre_transformed = pd.DataFrame(
            categorical_pipeline.transform(categorical_data).toarray(),
            columns = categorical_pipeline.get_feature_names_out()
        )
        numerical_pipeline = self.build_numerical_pipeline()
        data_transformed = pd.DataFrame(
            numerical_pipeline.fit_transform(data_pre_transformed),
            columns = categorical_pipeline.get_feature_names_out()
        )
        return data_transformed

    @staticmethod
    def build_categorical_pipeline() -> Pipeline:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                ("ohe", OneHotEncoder())
            ]
        )
        return categorical_pipeline

    def process_numerical_features(self, numerical_data: pd.DataFrame) -> pd.DataFrame:
        numerical_pipeline = self.build_numerical_pipeline()
        return pd.DataFrame(
            numerical_pipeline.fit_transform(numerical_data),
            columns=self.numerical_features
        )

    @staticmethod
    def build_numerical_pipeline() -> Pipeline:
        numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan)),
                ('scaler', StandardScaler())
            ]
        )
        return numerical_pipeline

    def fit(self):
        return self

    def transform(self) -> pd.DataFrame:
        categorical_part, numerical_part = self.X[self.categorical_features], self.X[self.numerical_features]
        categorical_part_processed = self.process_categorical_features(categorical_part)
        numerical_part_processed = self.process_numerical_features(numerical_part)
        return pd.concat([numerical_part_processed, categorical_part_processed], axis=1)
