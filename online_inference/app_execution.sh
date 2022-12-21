export MODEL_PATH="app/data/models/model.pkl"
echo MODEL_PATH variable is set to $MODEL_PATH

export TRANSFORMER_PATH="app/data/transformers/transformer.pkl"
echo TRANSFORMER_PATH variable is set to $TRANSFORMER_PATH


uvicorn app.main:app --host 0.0.0.0 --port 8000
