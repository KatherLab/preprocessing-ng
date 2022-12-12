FROM python
RUN apt-get update \
    && apt-get install -y python3-openslide python3-opencv \
    && pip install numpy tqdm openslide-python opencv-python fire
COPY . /preprocessing-ng
WORKDIR /preprocessing-ng