FROM python:3.12-slim

WORKDIR /app

RUN apt update && apt install git ffmpeg -y

RUN pip install -r https://raw.githubusercontent.com/Ioann44/webrtc-cv-public/refs/heads/master/resources/requirements.txt

ARG GIT_CLONE_INVALIDATE=none
RUN git clone https://github.com/Ioann44/webrtc-cv-public .
