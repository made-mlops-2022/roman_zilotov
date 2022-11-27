import os
import pickle
from fastapi import FastAPI
import pandas as pd

from app.entities import InputData

app = FastAPI()

model = None
transformer = None


@app.on_event('startup')
def get_model():
    path_to_model = os.getenv('MODEL_PATH')
    path_to_transformer = os.getenv('TRANSFORMER_PATH')

    with open(path_to_transformer, 'rb') as f:
        global transformer
        transformer = pickle.load(f)

    with open(path_to_model, 'rb') as f:
        global model
        model = pickle.load(f)


@app.get("/")
def root():
    return {'message': 'This is the start page! Go to /docs to try our model!'}


@app.get("/health")
def check_health():
    if model is not None and transformer is not None:
        return "The model is ready!"
    else:
        return "Something's wrong with the model!"


@app.post('/predict')
async def predict(input_data: InputData):
    transformed_input_data = transformer.transform(pd.DataFrame([input_data.dict()]))
    prediction = model.predict(transformed_input_data)
    prediction_verdict = 'the patient is healthy' if not prediction[0] else 'the patient is sick'
    return {'model prediction': prediction_verdict}


