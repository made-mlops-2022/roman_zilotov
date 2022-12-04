FROM python:3.8.15-slim-buster

COPY gen_data.py origin_data.csv requirements.txt /

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "gen_data.py"]