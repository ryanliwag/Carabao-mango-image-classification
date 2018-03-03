FROM python:3

MAINTAINER Ryan Joshua H. Liwag <rjhontomin@gmail.com>

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
COPY frozen_models/frozen_inference_graph.pb /app
COPY frozen_models/MTL_frozen_model.pb /app
COPY file.jpeg /app
COPY arial.ttf /app

RUN pip3 install --no-cache-dir -r requirements.txt

COPY predict_mango_classify.py /app

ENTRYPOINT ["python3", "./predict_mango_classify.py"]


