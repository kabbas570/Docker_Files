FROM continuumio/miniconda3
MAINTAINER hjkuijf

WORKDIR /usr/src/app

RUN pip install albumentations
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN pip install tqdm

RUN mkdir -p /output
COPY read_data1.py .
COPY models.py .
COPY val1.py .
COPY m_unet4.pth.tar .

CMD ["python", "./val1.py"]