import os
from src import PATH_TO_SRC

# MAKE CHANGES ONLY IN THE BOX BELOW
########################################
os.environ['KAGGLE_USERNAME'] = "romanzilotov"  # WRITE YOUR USERNAME
os.environ['KAGGLE_KEY'] = "94296465a88f22cd661536c996817286"  # WRITE YOUR KEY
########################################

from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    #print(os.environ.keys())
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        'cherngs/heart-disease-cleveland-uci',
        path=PATH_TO_SRC.joinpath('data/raw'),
        unzip=True
    )



if __name__ == '__main__':
    main()
