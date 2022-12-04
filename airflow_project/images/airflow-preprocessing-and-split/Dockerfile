FROM python:3.8.15-slim-buster

COPY preprocessing_and_split.py requirements.txt /

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "preprocessing_and_split.py"]