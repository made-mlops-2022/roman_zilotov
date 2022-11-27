import json
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    get_model()


def test_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': 'This is the start page! Go to /docs to try our model!'}


def test_predict():
    input_data = {
        "age": 50,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 1,
        "ca": 0,
        "thal": 0
        }
    response = client.post(
        '/predict',
        json.dumps(input_data)
    )
    assert response.status_code == 200
    assert response.json() == {'model prediction': 'the patient is healthy'}


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == "The model is ready!"


def test_missing_fields():
    input_data = {
        "age": 50,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 1,
        "ca": 0
    }
    response = client.post(
        '/predict',
        json.dumps(input_data)
    )
    assert response.status_code == 422
    message = 'Incorrect columns. Should be like'
    assert response.json()['detail'][0]['msg'][:len(message)] == message


def test_wrong_values():
    input_data = {
        "age": 500,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 1,
        "ca": 0,
        "thal": 0
    }
    response = client.post(
        '/predict',
        json.dumps(input_data)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == "Incorrect 'age' value (should be in [0, 125])"

