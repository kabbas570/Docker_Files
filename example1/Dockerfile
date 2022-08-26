FROM continuumio/miniconda3
MAINTAINER hjkuijf

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /output
COPY p1.py .

CMD ["python", "./p1.py"]