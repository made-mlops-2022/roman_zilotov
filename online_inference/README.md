**Homework 2**
==============================

## *Instructions*

### ***Building the image***
*From source*: go to `online_inference/` and run:
~~~
docker build -t romanzilotov/online_inference:v5 .
~~~
*From DockerHub*:
~~~
docker pull romanzilotov/online_inference:v5
~~~


### ***Running the container***

~~~
docker run -it —name online_inference -p 8000:8000 romanzilotov/online_inference:v5
~~~
After that the service will be running on  http://0.0.0.0:8000 

To make predictions go here and follow the instructions.


### ***Running tests***
Go to `online_inference/` and run:
~~~
sh tests_execution.sh
~~~

### ***Docker image size optimization***
1. Initial size: 1.31 Gb
2. Join two RUN commands into one in `Dockerfile` - *the size didn't change.*
3. Using `python:3.8.15-slim` instead of `python:3.8.15` - size changed to 562 Mb
4. Using `python:3.8.15-slim-buster` instead of `python:3.8.15-slim` - size changed to 555 Mb
