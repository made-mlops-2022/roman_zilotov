#1/bin/bash
export LOCAL_DATA_DIR=$(pwd)/data
export CUR_MODEL_DATE=2022-11-27
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build