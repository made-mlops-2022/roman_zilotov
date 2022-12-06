**Homework 1**
==============================

## *Instructions*

### ***Installation***
Go to `hw1_project/` and run:
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
~~~

### ***Loading the data***
1. Make changes to the file `hw1_project/src/data/data_loader.py` by typing your kaggle username and key.
2. Go back to `hw1_project/` and run:
    ~~~
    data_loader
    ~~~

### ***Training the model***
To use default parameters run:
~~~
python src/train_pipeline.py
~~~
Then the KNN will be used. If you want to use RandomForestClassifier run:
~~~
python src/train_pipeline.py model=rfc
~~~
By default grid_search will be applied. If you don't want to use it, run:
~~~
python src/train_pipeline.py model.grid_search=False
~~~

Fitted model will be saved to `hw1_project/models/model.pkl`.


### ***Making predictions***
To make predictions run:
~~~
python src/predict_pipeline.py
~~~
Predicted target will be saved to `hw1_project/data/predicted_target.txt`

You can also save target to a custom location using:
~~~
python src/predict_pipeline.py  -ppt=/Users/username/Desktop/predicted_target.txt
~~~

Or you can use your train data:
~~~
python src/predict_pipeline.py  -ptd=/Users/username/Desktop/data_train.txt
~~~

### ***Running tests***
1. To make synthetic data run:
   ~~~
   make_synthetic_data
   ~~~
   Data loads to `hw1_project/tests/synthetic_data/synthetic_data.csv`
2. To run tests run:
   ~~~
   python -m unittest tests/test_and_train_predict_pipeline.py
   ~~~


## *Project organization*
    ├── config
    │   ├── hydra            <- directory for logging config
    │   ├── model
    │   │     ├── knn.yaml   <- config for knn
    │   │     └── rfc.yaml   <- config for rfc
    │   └── config.yaml      <- general config
    │
    ├── data
    │   ├── processed            <- The final, canonical data sets for modeling.
    │   ├── raw                  <- The original, immutable data dump.
    │   └── predicted_target.txt <- file for predicted target by model
    │
    ├── models
    │   └── model.pkl            <- trained model
    │
    ├── notebooks
    │   └── 1.0-raz-initial-data-exploration.ipynb        <- basic EDA
    │
    ├── src
    │   ├── data
    │   │   ├── __init__.py
    │   │   └── data_loader.py               <- downloads data
    │   │
    │   ├── entities
    │   │   ├── __init__.py
    │   │   └── train_pipeline_params.py     <- sets params
    │   │
    │   ├── features
    │   │   ├── __init__.py
    │   │   └── build_features.py            <- prepares features(scaling and applying one hot encoder)
    │   │
    │   ├── models
    │   │   ├── __init__.py
    │   │   └── model_train.py               <- function for training
    │   │
    │   ├── __init__.py
    │   ├── train_pipeline.py                <- runs training pipeline
    │   └── predict_pipeline.py              <- runs predicting pipeline
    │
    ├── tests
    │   ├── synthetic_data
    │   │   ├── description.json
    │   │   ├── heatmap.png
    │   │   └── synthetic_data.csv
    │   │
    │   ├── make_synthetic_data.py                <- makes synthetic data
    │   └── test_and_train_predict_pipeline.py    <- runs tests
    │
    ├── README.md                <- Launch instructions and Project organization
    ├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    │
    └── setup.py                 <- makes project pip installable (pip install -e .) so src can be imported
